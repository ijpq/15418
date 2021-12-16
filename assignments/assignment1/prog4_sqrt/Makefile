CXX=g++ -m64 --std=c++11
# CXXFLAGS=-I../common -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -Iobjs/ -O3 -Wall
CXXFLAGS=-I../common -Iobjs/ -O3 -Wall

ISPC=ispc
# note: change target to avx for AVX capable machines
ISPCFLAGS=-O2 --target=avx2-i32x8 --arch=x86-64

OPENCV_INCLUDES=`pkg-config --cflags opencv4`
OPENCV_LIBS=`pkg-config --libs opencv4`
CVPLOT_INCLUDES=-I/home/tangke/15418/cv-plot-1.2.2/CvPlot/inc

APP_NAME=sqrt
OBJDIR=objs
COMMONDIR=../common

PPM_CXX=$(COMMONDIR)/ppm.cpp
PPM_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(PPM_CXX:.cpp=.o)))

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(OBJDIR)/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))

default: $(APP_NAME)

.PHONY: dirs clean

dirs:
		/bin/mkdir -p $(OBJDIR)/

clean:
		/bin/rm -rf $(OBJDIR) *.ppm *~ $(APP_NAME)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/sqrtSerial.o $(OBJDIR)/data.o $(OBJDIR)/sqrt_ispc.o $(PPM_OBJ) $(TASKSYS_OBJ)

$(APP_NAME): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDES) $(CVPLOT_INCLUDES) -o $@ $(OBJS) /home/tangke/opencv_contrib-master/modules/plot/samples/plot.o -lm $(TASKSYS_LIB) $(OPENCV_LIBS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $(OPENCV_INCLUDES) $(CVPLOT_INCLUDES) $< $(CXXFLAGS) $(OPENCV_LIBS) -c -o $@ 

$(OBJDIR)/%.o: $(COMMONDIR)/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/main.o: $(OBJDIR)/$(APP_NAME)_ispc.h $(COMMONDIR)/CycleTimer.h

$(OBJDIR)/%_ispc.h $(OBJDIR)//%_ispc.o: %.ispc
		$(ISPC) $(ISPCFLAGS) $< -o $(OBJDIR)/$*_ispc.o -h $(OBJDIR)/$*_ispc.h

