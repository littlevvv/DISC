#方法一：找出佔用該埠的程式並關閉


查詢佔用的程式
sudo lsof -i :5002

:::info
COMMAND     PID USER   FD   TYPE   DEVICE SIZE/OFF NODE NAME
python  3735017 r300    3u  IPv4 30813184      0t0  TCP *:5002 (LISTEN)
python  3738115 r300    3u  IPv4 30813184      0t0  TCP *:5002 (LISTEN)
python  3738115 r300    4u  IPv4 30813184      0t0  TCP *:5002 (LISTEN)

```
sudo netstat -tulnp | grep 5002
```


#方法二：使用 fuser 快速釋放埠號
```
sudo fuser -k 5002/tcp
```


#方法三：重啟 socketserver 前先檢查是否已啟動
若您是用 Python 啟動 socketserver，請確認是否已有執行中的實例。可在程式中加入錯誤處理：
```
import socketserver

try:
    server = socketserver.TCPServer(("localhost", 5002), Handler)
    server.serve_forever()
except OSError as e:
    print("埠號已被佔用:", e)

```
🧼 額外建議：避免重複啟動

若您使用 systemd 或 supervisor 管理服務，請確認該服務是否已啟動。
若您使用 Flask 或 FastAPI，請確認是否有背景執行的實例。