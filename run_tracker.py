import os
base_dir = "/home/pablo/DHARMA/berry_tracking/"
output_path = "output/modelo_reentrenado/"
input_path = "input/"
model_path = "2022.11.30_grapes_mix_iou.pth"

# Opening file
file1 = open('/home/pablo/DHARMA/berry_tracking/input/input.txt', 'r')
count = 0
  
# Using for loop
print("Using for loop")

for line in file1:
  line = line.replace("\n","")
  line2 = line.replace(".mp4", "")
  os.chdir(base_dir+output_path)
  try:
  	os.mkdir(line2)
  except:
  	pass
  os.chdir(base_dir)
  input_output = " --input "+input_path+line+" --output "+output_path+line2
  print(input_output)
  cmd = "python demo_with_tracker.py cdiou --arch hourglass --demo ../docs/demo.png --load_model "+model_path+"  -u '' -w '' --debug 0 --skip-frames 2 "+input_output
  os.system(cmd)
  
