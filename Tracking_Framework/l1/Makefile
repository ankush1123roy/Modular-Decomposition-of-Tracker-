all: IMGaffine.pyd APGLASSOup.pyd softresh.pyd

# ---- Link --------------------------- 
IMGaffine.pyd:  IMGaffine_c.o
	gcc -shared IMGaffine_c.o -o IMGaffine.pyd -L G:\Python27\libs -lpython27

APGLASSOup.pyd:  APGLASSOup.o
	gcc -shared APGLASSOup.o -o APGLASSOup.pyd -L G:\Python27\libs -lpython27
	
softresh.pyd : softresh_c.o
	gcc -shared softresh_c.o -o softresh.pyd -L G:\Python27\libs -lpython27
# ---- gcc C compile ------------------
IMGaffine_c.o:  IMGaffine_c.c C_arraytest.h
	gcc  -c -fPIC IMGaffine_c.c -I G:\Python27\include -I G:\Python27\Lib\site-packages\numpy\core\include
	
APGLASSOup.o:  APGLASSOup.c C_arraytest.h
	gcc  -c -fPIC APGLASSOup.c -I G:\Python27\include -I G:\Python27\Lib\site-packages\numpy\core\include
softresh_c.o: softresh_c.c softresh_arraytest.h
	gcc -c -fPIC softresh_c.c -I G:\Python27\include -I G:\Python27\Lib\site-packages\numpy\core\include