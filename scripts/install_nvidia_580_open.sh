#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "This installer must run as root." >&2
  exit 1
fi

readonly state_dir="/var/lib/codex-nvidia-580-update"
readonly log_file="/var/log/codex-nvidia-580-update.log"
mkdir -p "${state_dir}"
exec > >(tee -a "${log_file}") 2>&1

stage="preflight"
on_exit() {
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf '%s\n' "FAILED stage=${stage} status=${status}" | tee "${state_dir}/failed"
    systemctl start gdm.service 2>/dev/null || true
    systemctl start todeskd.service 2>/dev/null || true
  fi
}
trap on_exit EXIT

echo "=== NVIDIA 580 open driver update started $(date --iso-8601=seconds) ==="
echo "current_kernel=$(uname -r)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

stage="package-download"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get --download-only --yes install nvidia-driver-580-open

stage="stop-graphics"
systemctl stop todeskd.service 2>/dev/null || true
systemctl stop gdm.service
sleep 3

stage="remove-runfile-595"
/usr/bin/nvidia-uninstall --silent \
  --log-file-name="${state_dir}/nvidia-595-uninstall.log"

stage="install-580-open"
DEBIAN_FRONTEND=noninteractive apt-get --yes install nvidia-driver-580-open

stage="initramfs"
update-initramfs -u -k "$(uname -r)"

stage="verification"
dpkg-query -W -f='${Package} ${Version}\n' nvidia-driver-580-open
modinfo -F version nvidia
dkms status

date --iso-8601=seconds > "${state_dir}/installed-awaiting-reboot"
rm -f "${state_dir}/failed"
echo "Driver 580 installation completed; reboot scheduled in one minute."
shutdown -r +1 "NVIDIA 580 driver installation completed"
