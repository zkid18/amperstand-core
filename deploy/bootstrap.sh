#!/usr/bin/env bash
# bootstrap.sh — one-command setup for an Amperstand vault droplet.
#
# Run as root on a fresh Ubuntu 22.04+ box. Re-running is safe — every step
# checks before doing.
#
#   git clone https://github.com/zkid18/amperstand-core /opt/amperstand/amperstand-core
#   cd /opt/amperstand/amperstand-core
#   sudo bash deploy/bootstrap.sh
#
# What this does (in order):
#   1. apt install python3, caddy, ufw, etc.
#   2. Create the `amperstand` system user.
#   3. Generate AMPERSTAND_API_KEY (or keep existing).
#   4. Optionally prompt for OPENAI_API_KEY (Enter to skip).
#   5. Clone the amperstand CLI from GitHub (if absent).
#   6. Build venv, install amperstand-core + amperstand CLI editable.
#   7. Install + enable systemd units (server, vault-watcher, feed-sync.timer).
#      Email-watch unit is installed but stays disabled until you run
#      `sudo -u amperstand amperstand email setup` to configure IMAP creds.
#   8. Configure Caddy (http-only by default; bundled Caddyfile.tls for later).
#   9. Open ports 22/80/443 in UFW.
#  10. Start amperstand-server, wait for /health, print summary.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/amperstand}"
CORE_REPO="${REPO_ROOT}/amperstand-core"
CLI_REPO="${REPO_ROOT}/amperstand"
SERVICE_USER="amperstand"
DATA_DIR="/var/lib/amperstand/vault"
ENV_FILE="/etc/amperstand/env"
VENV_DIR="${REPO_ROOT}/venv"
PORT="${PORT:-8765}"
CLI_REPO_URL="${CLI_REPO_URL:-https://github.com/zkid18/amperstand-cli.git}"

if [ "$(id -u)" -ne 0 ]; then
    echo "error: must run as root (try: sudo bash $0)" >&2
    exit 1
fi

# ── 1. apt + caddy ─────────────────────────────────────────────────

echo "==> Updating apt and installing packages"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git curl ufw \
    debian-keyring debian-archive-keyring apt-transport-https

echo "==> Installing Caddy (if absent)"
if ! command -v caddy >/dev/null 2>&1; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
        | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
        | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    apt-get update -qq
    apt-get install -y -qq caddy
fi

# ── 2. service user ────────────────────────────────────────────────

echo "==> Creating service user '${SERVICE_USER}' (if absent)"
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin \
        --home-dir /var/lib/amperstand "$SERVICE_USER"
fi

# ── 3. data dir ────────────────────────────────────────────────────

echo "==> Preparing data dir ${DATA_DIR}"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$(dirname "$DATA_DIR")"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$DATA_DIR"

# ── 4. env file (AMPERSTAND_API_KEY + optional OPENAI_API_KEY) ──────

echo "==> Preparing env file ${ENV_FILE}"
install -d -m 0750 -o root -g "$SERVICE_USER" /etc/amperstand

if [ ! -f "$ENV_FILE" ]; then
    # Refuse to silently mint a NEW key when the vault already has data —
    # the operator probably restored from backup and forgot the env file,
    # not "this is a green-field install." Minting a new key in that case
    # silently rotates every existing client (CLI, bot, extension) with no
    # warning. Require an explicit confirmation.
    DATA_NONEMPTY=0
    if [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        DATA_NONEMPTY=1
    fi
    if [ "$DATA_NONEMPTY" = 1 ] && [ "${AMPERSTAND_BOOTSTRAP_FORCE_NEW_KEY:-}" != "1" ]; then
        echo
        echo "ERROR: ${ENV_FILE} is missing but ${DATA_DIR} is not empty." >&2
        echo "       Refusing to mint a new AMPERSTAND_API_KEY — that would silently" >&2
        echo "       rotate every client (CLI, bot, extension) against the existing vault." >&2
        echo
        echo "  If you restored from backup: restore the env file too (the deployment" >&2
        echo "  docs treat it as a separate backup line; it must come back with the vault)." >&2
        echo
        echo "  If you really do want a fresh key (this IS a clean install on top of" >&2
        echo "  stale data you'll discard), re-run with:" >&2
        echo "    AMPERSTAND_BOOTSTRAP_FORCE_NEW_KEY=1 sudo bash deploy/bootstrap.sh" >&2
        echo
        exit 2
    fi

    KEY="$(openssl rand -hex 32)"
    cat > "$ENV_FILE" <<EOF
AMPERSTAND_API_KEY=${KEY}
AMPERSTAND_DATA_DIR=${DATA_DIR}
EOF
    chmod 0640 "$ENV_FILE"
    chown root:"$SERVICE_USER" "$ENV_FILE"
    echo
    # Only echo the key to stdout on an interactive TTY. On non-interactive
    # runs (CI provisioner, `tee bootstrap.log`, terraform remote-exec, etc.)
    # the operator's "logfile" would otherwise capture the master credential
    # in a less-restrictive location than /etc/amperstand/env (0640 root:amp).
    if [ -t 1 ]; then
        echo "    NEW API KEY GENERATED — store this somewhere safe NOW."
        echo "    AMPERSTAND_API_KEY=${KEY}"
        echo
    else
        echo "    NEW API KEY GENERATED (suppressed from non-interactive stdout)."
        echo "    Retrieve it on the server with:"
        echo "      grep '^AMPERSTAND_API_KEY=' ${ENV_FILE} | cut -d= -f2-"
        echo
    fi
else
    echo "    env file already exists, leaving it alone"
fi

# Prompt for OPENAI_API_KEY if not set. Skipping is fine — semantic search,
# hybrid search, rerank, /chat, and audio-fallback transcription will all
# respond 503 until you set it. The README has the cost breakdown.
if ! grep -q '^OPENAI_API_KEY=' "$ENV_FILE"; then
    echo
    echo "==> OpenAI API key (optional — enables /chat, semantic + hybrid search, rerank)"
    echo "    Without it, those endpoints return 503. BM25 search via /vault/search still works."
    echo "    Paste your key now, or press Enter to skip and add later by editing ${ENV_FILE}."
    if [ -t 0 ]; then
        # Interactive run — actually prompt.
        printf "    OPENAI_API_KEY (sk-...): "
        read -r OPENAI_KEY_INPUT
        if [ -n "${OPENAI_KEY_INPUT:-}" ]; then
            printf '\nOPENAI_API_KEY=%s\n' "$OPENAI_KEY_INPUT" >> "$ENV_FILE"
            echo "    OPENAI_API_KEY added to ${ENV_FILE}."
        else
            echo "    Skipped — /chat + semantic + hybrid will 503 until you set it."
        fi
    else
        # Non-interactive (e.g. piped install) — skip silently.
        echo "    non-interactive run — skipped. Add OPENAI_API_KEY to ${ENV_FILE} later."
    fi
fi

# Pre-write the CLI's vault-backend config so `sudo -u amperstand amperstand …`
# works out of the box. Without this, the very first command in the README
# (`amperstand capture <url>`) errors with "no vault backend configured" because
# the CLI doesn't know it should talk to the local server. The file goes into
# the amperstand user's home (~/.amperstand/config.json) and points at loopback
# + reads the API key from the env file that systemd loads.
echo "==> Writing CLI vault-backend config for ${SERVICE_USER}"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0700 \
    /var/lib/amperstand/.amperstand
cat > /var/lib/amperstand/.amperstand/config.json <<EOF
{
  "vault": {
    "backend": {
      "kind": "http",
      "http": {
        "url": "http://127.0.0.1:${PORT}",
        "api_key_env": "AMPERSTAND_API_KEY"
      }
    }
  }
}
EOF
chown "$SERVICE_USER":"$SERVICE_USER" /var/lib/amperstand/.amperstand/config.json
chmod 0600 /var/lib/amperstand/.amperstand/config.json

# ── 5. clone CLI repo ──────────────────────────────────────────────

echo "==> Fetching amperstand CLI from ${CLI_REPO_URL}"
if [ ! -d "$CLI_REPO/.git" ]; then
    git clone --quiet "$CLI_REPO_URL" "$CLI_REPO"
else
    echo "    CLI repo already present at ${CLI_REPO}"
fi

# ── 6. venv + editable installs ────────────────────────────────────

echo "==> Building venv at ${VENV_DIR}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -e "$CORE_REPO"
# Playwright only drives the Chromium build that matches its own version,
# and pip upgrades playwright without fetching it. Without this step every
# LinkedIn capture fails with 422. Runs as the service user so the browser
# lands in its own cache (~/.cache/ms-playwright), where the server looks.
runuser -u "$SERVICE_USER" -- env HOME=/var/lib/amperstand \
    "${VENV_DIR}/bin/playwright" install chromium
"${VENV_DIR}/bin/pip" install --quiet -e "$CLI_REPO"

echo "==> Setting ownership on ${REPO_ROOT}"
chown -R "${SERVICE_USER}":"${SERVICE_USER}" "${REPO_ROOT}"

# ── 7. systemd units ───────────────────────────────────────────────

echo "==> Installing systemd units"
for unit in \
    amperstand-server.service \
    amperstand-vault-watcher.service \
    amperstand-feed-sync.service \
    amperstand-feed-sync.timer \
    amperstand-email-watch.service
do
    install -m 0644 "${CORE_REPO}/deploy/systemd/${unit}" "/etc/systemd/system/${unit}"
done
systemctl daemon-reload

echo "==> Enabling units (server, vault-watcher, feed-sync.timer)"
systemctl enable amperstand-server.service
systemctl enable amperstand-vault-watcher.service
systemctl enable amperstand-feed-sync.timer

# Email-watch deliberately left disabled — needs IMAP creds first.
# User enables manually after running `amperstand email setup`.

# ── 8. caddy ───────────────────────────────────────────────────────

echo "==> Configuring Caddy (HTTP-only default)"
if [ ! -f /etc/caddy/Caddyfile.bootstrap-backup ] && [ -f /etc/caddy/Caddyfile ]; then
    cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bootstrap-backup
fi
install -m 0644 "${CORE_REPO}/deploy/Caddyfile.no-tls" /etc/caddy/Caddyfile
systemctl reload caddy || systemctl start caddy

echo "    Caddyfile.tls (with auto-HTTPS) is at ${CORE_REPO}/deploy/Caddyfile.tls"
echo "    Swap it in after pointing a domain at this box."

# ── 9. firewall ────────────────────────────────────────────────────

echo "==> Configuring firewall (UFW)"
ufw allow 22/tcp >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
ufw --force enable >/dev/null

# ── 10. start + verify ─────────────────────────────────────────────

echo "==> Starting amperstand-server"
systemctl restart amperstand-server.service
systemctl start amperstand-vault-watcher.service
systemctl start amperstand-feed-sync.timer

echo "==> Waiting for /health (up to 30s)"
for i in $(seq 1 30); do
    if curl -fsS -o /dev/null "http://127.0.0.1:${PORT}/health"; then
        echo "    healthy after ${i}s"
        break
    fi
    sleep 1
    if [ "$i" = "30" ]; then
        echo "    /health didn't come up in 30s — check: sudo journalctl -u amperstand-server -n 50"
        exit 1
    fi
done

echo
echo "✓ Bootstrap complete."
echo
echo "  Public URL:  http://$(curl -fsS https://ipv4.icanhazip.com 2>/dev/null || echo "this-droplet"):80/health"
echo "  Local:       curl -s http://127.0.0.1:${PORT}/health"
echo "  Logs:        sudo journalctl -u amperstand-server -f"
echo "  Admin:       sudo -u ${SERVICE_USER} ${VENV_DIR}/bin/amperstand-admin stats"
echo
echo "  Optional:"
echo "    Email watcher (newsletters → vault):"
echo "      sudo -u ${SERVICE_USER} ${VENV_DIR}/bin/amperstand email setup"
echo "      sudo systemctl enable --now amperstand-email-watch"
echo
echo "    TLS / domain (when you have one):"
echo "      sudo cp ${CORE_REPO}/deploy/Caddyfile.tls /etc/caddy/Caddyfile"
echo "      sudo \$EDITOR /etc/caddy/Caddyfile     # replace YOUR_DOMAIN_HERE"
echo "      sudo systemctl reload caddy"
echo
