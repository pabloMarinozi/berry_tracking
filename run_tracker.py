import os
base_dir = "/mnt/datos/onedrive/Doctorado/3df/berry_tracking/"
output_path = "/mnt/datos/capturas/videos_pablo_2023_02_08/output/"
input_path = "/mnt/datos/capturas/videos_pablo_2023_02_08/"
model_path = "/mnt/datos/models/2022.11.30_grapes_mix_iou.pth"

# Opening file
file1 = open('/mnt/datos/capturas/videos_pablo_2023_02_08/input.txt', 'r')
count = 0

# Using for loop
print("Using for loop")

for line in file1:
  line = line.replace("\n","")
  line2 = line.replace(".mp4", "")
  os.chdir(output_path)
  try:
  	os.mkdir(line2)
  except:
  	pass
  os.chdir(base_dir)
  input_output = " --input "+input_path+line+" --output "+output_path+line2
  print(input_output)
  cmd = "python demo_with_tracker.py cdiou --arch hourglass --demo ../docs/demo.png --load_model "+model_path+"  -u '' -w '' --debug 0 --skip-frames 1 "+input_output
  os.system(cmd)
  processed_files = open(input_path + 'processed.txt', 'a')
  processed_files.write(line)
  processed_files.close()