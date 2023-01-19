## YOLOv6-Count_Objects


# Sample photos

![image2](https://user-images.githubusercontent.com/40724187/213548405-f26dfea8-13e3-4725-b519-9c1eb4bbbc7d.jpg)
![image3](https://user-images.githubusercontent.com/40724187/213548411-ab74ad2a-7af1-45af-b05e-10923e50f95b.jpg)


# Sample video

https://user-images.githubusercontent.com/40724187/213550223-9393092a-96d8-49cd-97da-9c4f41ebf42f.mp4

# Steps to implement
1. Create virtual environment
2. Clone the [repo](https://github.com/onyekaokonji/YOLOv6_Object_Count)
3. Run "pip install -r requirements.txt"
4. Having downloaded the model checkpoint of choice from [here](https://github.com/meituan/YOLOv6/releases/tag/0.3.0), Run "python tools/infer.py --weights yolov6s.pt --source img.jpg" while feeding the necessary arguments as needed.
