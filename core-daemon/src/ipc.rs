//! Newline-delimited JSON-RPC 2.0 over a Unix domain socket.
//!
//! Used for CLI ↔ daemon and daemon ↔ AI engine (Python). Unix sockets keep
//! everything local, avoid a TCP port, and let us restrict access with `0600`
//! file permissions.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as TokioBufReader};
use tokio::net::UnixListener;
use tokio::sync::{mpsc, oneshot};

// ---------------------------------------------------------------------------
// Server side
// ---------------------------------------------------------------------------

pub struct Request {
    pub method: String,
    pub params: Value,
    pub reply: oneshot::Sender<Result<Value, String>>,
}

pub type RequestSender = mpsc::UnboundedSender<Request>;

/// Bind the socket and serve until the process exits.
pub async fn serve(socket: PathBuf, tx: RequestSender) -> Result<()> {
    if socket.exists() {
        let _ = std::fs::remove_file(&socket);
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener =
        UnixListener::bind(&socket).with_context(|| format!("bind {}", socket.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600));
    }
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let tx = tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = connection(stream, tx).await {
                        if !is_disconnect(&e) {
                            eprintln!("[ipc] connection error: {e}");
                        }
                    }
                });
            }
            Err(e) => {
                eprintln!("[ipc] accept: {e}");
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

fn is_disconnect(e: &anyhow::Error) -> bool {
    let text = format!("{e:#}");
    text.contains("Broken pipe")
        || text.contains("Connection reset")
        || text.contains("UnexpectedEof")
        || text.contains("EOF")
}

async fn connection(stream: tokio::net::UnixStream, tx: RequestSender) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = TokioBufReader::new(read_half);
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            return Ok(());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {"code": -32700, "message": format!("parse error: {e}")}
                });
                write_half.write_all(format!("{err}\n").as_bytes()).await?;
                write_half.flush().await?;
                continue;
            }
        };
        let id = parsed.get("id").cloned().unwrap_or(Value::Null);
        let method = parsed
            .get("method")
            .and_then(|m| m.as_str())
            .unwrap_or_default()
            .to_string();
        let params = parsed
            .get("params")
            .cloned()
            .unwrap_or_else(|| Value::Object(Default::default()));

        let (rtx, rrx) = oneshot::channel();
        if tx
            .send(Request {
                method,
                params,
                reply: rtx,
            })
            .is_err()
        {
            bail!("core request channel closed");
        }
        let response = match rrx.await {
            Ok(Ok(result)) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Ok(Err(message)) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32000, "message": message}
            }),
            Err(_) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32001, "message": "core dropped the request"}
            }),
        };
        write_half
            .write_all(format!("{response}\n").as_bytes())
            .await?;
        write_half.flush().await?;
    }
}

// ---------------------------------------------------------------------------
// Client side (blocking — used by the CLI and by the daemon→AI bridge)
// ---------------------------------------------------------------------------

struct Conn {
    writer: UnixStream,
    reader: BufReader<UnixStream>,
}

pub struct RpcClient {
    socket: PathBuf,
    timeout: Duration,
    conn: Mutex<Option<Conn>>,
    next_id: AtomicU64,
}

impl RpcClient {
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self::with_timeout(socket, Duration::from_secs(2))
    }

    pub fn with_timeout(socket: impl Into<PathBuf>, timeout: Duration) -> Self {
        Self {
            socket: socket.into(),
            timeout,
            conn: Mutex::new(None),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Is the peer accepting connections?
    pub fn reachable(&self) -> bool {
        UnixStream::connect(&self.socket).is_ok()
    }

    pub fn call(&self, method: &str, params: Value) -> Result<Value> {
        let mut last_err = None;
        for attempt in 0..2 {
            match self.try_call(method, &params) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    last_err = Some(e);
                    // Drop the connection so the next attempt reconnects.
                    *self.conn.lock().unwrap() = None;
                    if attempt == 0 {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("rpc failed")))
    }

    fn try_call(&self, method: &str, params: &Value) -> Result<Value> {
        let mut guard = self.conn.lock().unwrap();
        if guard.is_none() {
            let stream = UnixStream::connect(&self.socket)
                .with_context(|| format!("connect {}", self.socket.display()))?;
            stream.set_read_timeout(Some(self.timeout))?;
            stream.set_write_timeout(Some(self.timeout))?;
            let reader_stream = stream.try_clone()?;
            reader_stream.set_read_timeout(Some(self.timeout))?;
            *guard = Some(Conn {
                writer: stream,
                reader: BufReader::new(reader_stream),
            });
        }
        let conn = guard.as_mut().expect("connection just created");
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        conn.writer.write_all(format!("{request}\n").as_bytes())?;
        conn.writer.flush()?;
        let mut line = String::new();
        let n = conn.reader.read_line(&mut line)?;
        if n == 0 {
            bail!("peer closed the connection");
        }
        let response: Value = serde_json::from_str(line.trim())?;
        if let Some(err) = response.get("error") {
            let message = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("rpc error");
            bail!("{message}");
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn roundtrip_over_unix_socket() {
        let dir = std::env::temp_dir().join(format!("mext-ipc-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket = dir.join("t.sock");
        let (tx, mut rx) = mpsc::unbounded_channel::<Request>();
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let out = match req.method.as_str() {
                    "ping" => Ok(json!("pong")),
                    "add" => {
                        let a = req.params.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                        let b = req.params.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                        Ok(json!(a + b))
                    }
                    other => Err(format!("unknown method {other}")),
                };
                let _ = req.reply.send(out);
            }
        });
        let sock = socket.clone();
        tokio::spawn(async move {
            let _ = serve(sock, tx).await;
        });
        tokio::time::sleep(Duration::from_millis(120)).await;

        let started = Instant::now();
        let client = RpcClient::new(socket);
        assert!(client.reachable());
        assert_eq!(client.call("ping", json!({})).unwrap(), json!("pong"));
        assert_eq!(
            client.call("add", json!({"a": 20, "b": 22})).unwrap(),
            json!(42)
        );
        assert!(client.call("nope", json!({})).is_err());
        assert!(started.elapsed() < Duration::from_secs(2));
        std::fs::remove_dir_all(&dir).ok();
    }
}
