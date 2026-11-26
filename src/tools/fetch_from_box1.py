#!/usr/bin/env python3
"""机器间直连取数据：在新 4090 上运行，从原 4090 拉取 data_pack.tgz

避免经过本地 Mac 中转（实测中转只有 86KB/s，71MB 要 14 分钟）。
AutoDL 内网同区互通，SCP 通常能到 10~50 MB/s。

用法（在新机执行）：
    python fetch_from_box1.py
"""
import os
import sys

import paramiko

SRC_HOST = "connect.nmb2.seetacloud.com"
SRC_PORT = 24865
SRC_USER = "root"
SRC_PASS = os.environ.get("BOX1_PASS", "")
SRC_FILE = "/tmp/data_pack.tgz"
DST = "/root/autodl-tmp/data_pack.tgz"


def main():
    if not SRC_PASS:
        print("需要 BOX1_PASS 环境变量（原机 root 密码）")
        sys.exit(1)

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"连接 {SRC_HOST}:{SRC_PORT} …")
    cli.connect(SRC_HOST, port=SRC_PORT, username=SRC_USER, password=SRC_PASS,
                timeout=30, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()
    st = sftp.stat(SRC_FILE)
    print(f"源文件 {SRC_FILE} = {st.st_size/1e6:.1f} MB")

    done = [0]

    def cb(a, b):
        done[0] = a
        pct = a / b * 100 if b else 0
        sys.stdout.write(f"\r  {a/1e6:.1f}/{b/1e6:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()

    print("开始传输 …")
    sftp.get(SRC_FILE, DST, callback=cb)
    sftp.close()
    cli.close()
    print(f"\n完成 -> {DST}  ({os.path.getsize(DST)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
