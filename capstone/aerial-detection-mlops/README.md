# aerial-detection-mlops
A complete end-to-end MLOps pipeline used to build, deploy, monitor, improve, and scale a YOLOv7-based aerial object detection model
 
 If you want to run it, go to the OHIO region of AWS and spin up Instance ID: i-0bff252a9610efdf7   (aerial-detection-mlops-001)  
 
 ssh in,`source activate pytorch` 
 
 `uvicorn main:app --host 0.0.0.0 --port 8000`
 
 head over to the FastAPI endpoint and upload an image then run inference on it, should work
 
 basically if you read the `app.py` you'll see that the endpoint will POST an image (and hardcode it with the name `inference/images/horses.jpg'``, dumb name, long story, it's late at night!)  
 
 Then `detect.detect()` will run the YOLOv7 detect function (hardcoded stuff in there too if you look) on that file in that location and return the image w/bounding boxes n probs on it.  
 
 Need to make it flexible and maybe keep images in memory instead of writing them to disk. Also nothinng going on with Docker in this. Lots of stuff to change. Also, we should move it to a cheaper box. The only thing with that is that right now the code expects a GPU at slot 0 and to switch it up will be kinda a pain.  Also, we will need a fat box at some point for transfer learning anyway, so not sure what the plan will be. 
 
