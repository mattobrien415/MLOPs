# aerial-detection-mlops
A complete end-to-end MLOps pipeline used to build, deploy, monitor, improve, and scale a YOLOv7-based aerial object detection model
 
 If you want to run it, go to the OHIO region of AWS and spin up Instance ID: i-0bff252a9610efdf7   (aerial-detection-mlops-001)  
 
 ssh in,`source activate pytorch` 
 
 `uvicorn main:app --host 0.0.0.0 --port 8000`
 
 Head over to the public endpoint and it takes 2 steps. First, upload an image then run inference on it, should work.
 
Basically if you read the `app.py` it has those 2 steps implemented there. You'll see that the endpoint will POST an image (and hardcode it with the name `inference/images/horses.jpg`, dumb name, long story, it's late at night!). It literally gets written to the disk.  
 
Then `detect.detect()` will run the YOLOv7 detect function (hardcoded stuff in that file too if you look) on the `horses.jpg` (dumb name I know I know) file in that location that you just put there from the previous step. It will return the image w/bounding boxes n probs on it.  
 
 Need to make it flexible and maybe keep images in memory using IO library instead of writing (and overwriting) things to disk. Also nothinng going on with Docker in this. Lots of stuff to change. Also, we should move it to a cheaper box. The best bet would be to just spin up the same community AMI (cos it has that super magical `pytorch` virtualenv, but with one GPU instead of the 4 or whatever this one has (!)
 
