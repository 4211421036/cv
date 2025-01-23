import asyncio
import websockets
import cv2
import base64
import numpy as np

async def websocket_handler(websocket, path):
    try:
        async for message in websocket:
            # Decode incoming frame
            img_data = base64.b64decode(message.split(',')[1])
            img = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Encode and send processed frame
            _, buffer = cv2.imencode('.jpg', img)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(f"data:image/jpeg;base64,{processed_frame}")

    except Exception as e:
        print(f"WebSocket error: {e}")

async def main():
    server = await websockets.serve(websocket_handler, "0.0.0.0", 8089)
    print("WebSocket server started")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
