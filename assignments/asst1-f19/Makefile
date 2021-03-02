
FILES= prog1_mandelbrot_threads/mandelbrot.cpp \
	prog2_vecintrin/functions.cpp \
	prog3_mandelbrot_ispc/mandelbrot.ispc \
	prog4_sqrt/data.cpp \
	prog5_saxpy/saxpyStreaming.cpp

handin.tar: $(FILES)
	tar cvf handin.tar $(FILES)

clean:
	(cd prog1_mandelbrot_threads; make clean)
	(cd prog2_vecintrin; make clean)
	(cd prog3_mandelbrot_ispc; make clean)
	(cd prog4_sqrt; make clean)
	(cd prog5_saxpy; make clean)
	rm -f *~ handin.tar
