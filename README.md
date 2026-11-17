# **Features of the CCTV AI Analytics System**

**CCTV AI Analytics System** provides several features for monitoring vehicles and people through a camera in real time.

## **1. Main Features**

[Vehicle Classifier | CVEDIA | AI Video Analytics for any hardware](https://images.openai.com/static-rsc-4/aNLwpPqYCVKKPVip8XqFhXzhn2hDA34bTsKrahj5rFk0FtZKqaZUniem-NAqMELfoRwh8UDWCZGP_fHUrmXzn07d1UOawp5c9qxaevqcY0aMSXwlbt3BSdMi4T9Y5BOYZsn7PxeYWjKK2H_3dEh84ELgU75RE_qv8_IfCxxvbF8?purpose=inline)

**1. Vehicle Detection and Counting**

- Automatically detects cars and motorcycles.
- Counts vehicles passing through the camera's field of view.
- Tracks vehicle movements to prevent duplicate counting.
- Displays detection bounding boxes on the video feed.

[Enhancing Business Security With PIAM: How i3 AI Sentry Is Unlocking New Potential in Physical Identity Access Management Systems](https://images.openai.com/static-rsc-4/T01GSpZ7rbC44c_L4T-GB9L_UbRcJMbcR0pYoBchR2cC7ZMZGWXQT3dtg0B9e_dytZ_gBkG0i2ofuTOksnd70Rr9vmLIQlaJ5Zjpt29lMsZIPKPyQZJ7_KwPCO8j47ppRSeJdi077y5iLHNknwOnKPh4aAkpJOzVbPi0Im8NnDM?purpose=inline)

**2. Face Recognition**

- Detects human faces in front of the camera.
- Recognizes faces that have been registered.
- Displays the person's name when a recognized face is detected.
- Displays **Unknown** for unrecognized faces.
- Records a person's presence based on their appearance.

[ANPR Parking: Smart, Fast, and Secure Parking for Modern Cities](https://images.openai.com/static-rsc-4/rO37ATQyb_f6EPHYtcohQgDcXipAcsNMGk6nJmeSNMqvBMmyW01aqdtwWuFlIwq5_wteDOpPXKUp4jPcrlg_czgmJgTv9fVEscEl6sCL6vbLb2v6-w4lxZLvA8UMnHN6qvpfzvbJQzV7R6InffxJ9H4UmV_WCXYIftNFOvcHBUY?purpose=inline)

**3. License Plate Recognition (OCR)**

- Extracts the vehicle image region.
- Detects and reads license plate characters using OCR.
- Stores the recognition results in the vehicle event logs.
- Displays `???` when the reading process fails.

[Sunell](https://images.openai.com/static-rsc-4/iDh1DbtLHSBIl6a0a2HD9tH4T7rfWIvTmkHgztJGlEm5HyfzhJ-08hd8z0ShSBs9eAfN2MYQyPj6GbOg6u0mQbY1y3vrQ0bUctIqUwxqYzgL6UIjLgsmr7SpOKEQRQFqgXwt9gNm7fdxaD_A3TxC7gylyA7nGRcC6a8n2FN3ybo?purpose=inline)

**4. Virtual Line Counting**

- Allows users to create virtual lines on the camera view.
- Detects vehicles crossing the virtual lines.
- Identifies the direction of movement: **Entry** and **Exit**.
- Supports multiple virtual lines with different names.
- Allows users to reverse the line direction.

[AI-powered analytics | Axis Communications](https://images.openai.com/static-rsc-4/awZ6uWMZuSORcmWrdlDuUrzRNUAEDYCXrEWOCv0vsuIx8_2rGfSOUw-qa3AITYDA9eIkL36QUfVQ8M-Cd-Ef3NX2KPPJsU6JRtmnZPIdQY-UlJHvOvLMwi3dDy-86h1qkSAFUKnybGtiQLw5ppWbmw-dqgbvKUPx0fA4QyVnb3U?purpose=inline)

**5. Dashboard and Statistics**

- Displays vehicle and person statistics.
- Shows daily and overall activity counts.
- Displays event history.
- Shows camera and detection system status.
- Displays video FPS and detection FPS.

## **2. Data Management Features**

In addition to AI-powered features, the application provides the following data management capabilities:

| Feature               | Description                                                                  |
| --------------------- | ---------------------------------------------------------------------------- |
| Face Management       | Register, rename, and delete registered faces.                               |
| Event Log Filtering   | Filter events by category, vehicle type, direction, and date.                |
| Event Log Search      | Search event records using keywords.                                         |
| License Plate Editing | Correct incorrectly recognized license plate numbers.                        |
| Event Log Deletion    | Delete individual events or clear all logs by category.                      |
| CSV Export            | Export event history for further processing in Microsoft Excel.              |
| Manual Data Entry     | Manually record vehicle type, direction, quantity, license plate, and notes. |

## **3. Camera Configuration Features**

The application supports two types of camera sources:

- **USB Camera:** Uses a laptop webcam or an external USB camera.
- **IP Camera:** Connects to network cameras using RTSP, RTSPS, HTTP, HTTPS, or RTMP protocols.

Camera configuration also includes:

- Scanning for available camera devices.
- Selecting a USB camera device index.
- Selecting a camera capture backend: Auto, DirectShow, or Media Foundation (Windows).
- Adjusting video resolution and FPS.
- Testing the camera connection before use.

## **4. AI and Performance Settings**

Users can customize the system's performance through several configuration options:

- **Quality Preset:** Auto, Power Saving, Balanced, Accurate, and Custom.
- **YOLO Image Size:** Adjusts the image processing size for vehicle detection.
- **Vehicle Sensitivity:** Controls vehicle detection sensitivity.
- **Face Sensitivity:** Controls face detection sensitivity.
- **Face Matching Tolerance:** Adjusts the tolerance level for face matching.
- **Face Detection Interval:** Controls how frequently face detection is performed.
- **Processing Device:** Selects CPU, GPU, or Auto.
- **View Mode:** Selects Parking, Face, All, or Clean display modes.

## **5. AI Technology Overview**

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| YOLOv8n          | Vehicle detection                         |
| Vehicle Tracker  | Tracks vehicle movements                  |
| YuNet            | Detects face locations                    |
| Face Recognition | Identifies registered faces               |
| EasyOCR          | Reads vehicle license plates              |
| OpenCV           | Processes images and video streams        |
| SQLite           | Stores application data and event history |
| FastAPI          | Runs the application's backend            |

## **Conclusion**

The **CCTV AI Analytics System** is more than a conventional CCTV application. It is an AI-powered surveillance system that integrates vehicle detection, vehicle entry and exit counting, face recognition, license plate recognition, activity logging, and data management into a single platform.

**Technical note:** License plate recognition depends on EasyOCR being available and functioning correctly, as well as the quality of the camera image. If the OCR module is unavailable or fails to initialize, the application cannot provide normal license plate recognition results.
