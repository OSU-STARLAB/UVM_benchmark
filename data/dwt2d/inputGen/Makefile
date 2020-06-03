CXX := /usr/bin/g++
INCLUDES := -I. 
COMMONFLAGS += $(INCLUDES) 

CXXOPTIONS = -std=c++11 -O2
CXXFLAGS += $(COMMONFLAGS) $(CXXOPTIONS)

LINK := $(CXX)

TARGET = create.exe
OBJS =  bmp.c.o create.c.o

LINKLINE = $(LINK) -o $(TARGET) $(OBJS) 


$(TARGET):  $(OBJS) 
	$(LINKLINE)


%.c.o: %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	rm -rf $(TARGET) *.o 

