//! `mextd` — thin daemon wrapper (same code path as `mext start`), convenient
//! for launchd / `nohup` use where a subcommand is awkward.

use clap::Parser;
use mext_core::run::{self, StartArgs};

#[derive(Parser)]
#[command(
    name = "mextd",
    version,
    about = "MEXT-M1 daemon (identical to `mext start`)"
)]
struct Cli {
    #[command(flatten)]
    start: StartArgs,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = cli.start.to_config()?;
    run::daemon(config).await
}
