# randaugment
an personal  implementation  about randaugment (testing accuracy in  cifar-10) 

This repo  a  partly  copy  from   https://github.com/DeepVoltaire/AutoAugment  ,it  a  implemetation  about  autoaugment  published by google .  I  just change some  code  about  operations  selected  way.


I  must  say  a  very  important  point here , there is  no  need  to   set  the  param magnitude  as  same  as  google's ,  there  is  different best  max  magnitude  for  different  dataset ,  so  I  wrote  a  test fun  for  you  to  test   every  single   operation  with different  magnitude.



# 1： if you  want to  just  call  this  class ,write  like above , this  is  for  call the whole fun



img_augment = Rand_Augment()   
img_origal = Image.open(r'0a38b552372d.png')  
img_final = img_augment(img_origal)   
plt.imshow(img_final)  
plt.show()  
print('how to  call')  

# 2: if you  want  to test  a single fun,write  like  above 

img_augment = Rand_Augment()  
img_origal = Image.open(r'0a38b552372d.png')  
img_final = img_augment.test_single_operation(img_origal, 'shearX')  
plt.imshow(img_final)  
plt.show()  
print('how  to test')  


Actually ,this  is  my  first  time to write a   reposity in Github ,  if there is  some bug  here , just  point  it directly. I'm  almost  in the  internet  every  day  :)
