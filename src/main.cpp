#include "mujoco/mujoco.h"
#include "glfw/glfw3.h"
#include "stdio.h"
#include <iostream>
#include <array>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct output{
  std::array<double,18>state;
  float reward;
  bool terminated;
  bool truncated;
};

char error[1000];
mjModel* m;
mjData* d;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

GLFWwindow* window = nullptr;

bool lPressed = false;
bool shiftPressed = false;
double oldx = 0;
double oldy = 0;

static void mouseCallback(GLFWwindow* window, double xpos, double ypos){
  if(shiftPressed && lPressed){
    mjv_moveCamera(m,mjMOUSE_MOVE_V,0*(xpos-oldx),0.001*(ypos-oldy),&scn,&cam);
    mjv_moveCamera(m,mjMOUSE_MOVE_H,0.001*(xpos-oldx),0*(ypos-oldy),&scn,&cam);
    oldx = xpos;
    oldy = ypos;
  }
  else if(lPressed){
    mjv_moveCamera(m,mjMOUSE_ROTATE_V,0.001*(xpos-oldx),0*(ypos-oldy),&scn,&cam);
    mjv_moveCamera(m,mjMOUSE_ROTATE_H,0*(xpos-oldx),0.001*(ypos-oldy),&scn,&cam);
    oldx = xpos;
    oldy = ypos;
  }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
  if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
    lPressed = true;
    glfwGetCursorPos(window,&oldx,&oldy);
  }
  if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
    lPressed = false;
  }
}

static void buttonCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(action == GLFW_PRESS && key == GLFW_KEY_LEFT_SHIFT){
      shiftPressed = true;
      glfwGetCursorPos(window,&oldx,&oldy);
    }
    if(action == GLFW_RELEASE && key == GLFW_KEY_LEFT_SHIFT){
      shiftPressed = false;
    }
}

std::array<double,18> sim_init(){
  m = mj_loadXML("crow.xml", NULL, error, 1000);
  if (!m) {
    printf("%s\n", error);
  }
  
  // make data corresponding to model
  d = mj_makeData(m);
  
  
  if(!glfwInit()){
    std::cout<<"window creation failed";
  }
  window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  
  glfwSetCursorPosCallback(window,mouseCallback);
  glfwSetMouseButtonCallback(window,mouse_button_callback);
  glfwSetKeyCallback(window,buttonCallback);
  
  mjv_defaultFreeCamera(m,&cam);
  mjv_defaultOption(&opt);
  mjr_defaultContext(&con);
  
  mjv_makeScene(m, &scn, 1000);
  mjr_makeContext(m, &con, mjFONTSCALE_100);
  
  double camConfig[] = {90,-45,4,0,0,0};
  
  cam.azimuth = camConfig[0];
  cam.elevation = camConfig[1];
  cam.distance = camConfig[2];
  cam.lookat[0] = camConfig[3];
  cam.lookat[1] = camConfig[4];
  cam.lookat[2] = camConfig[5];

  mjtNum* qpos = d->qpos;
  return {d->qpos[2],0,0,0,d->qpos[7],d->qpos[8],d->qpos[9],d->qpos[10],d->qpos[11],d->qvel[2],d->qvel[3],d->qvel[4],d->qvel[5],d->qvel[6],d->qvel[7],d->qvel[8],d->qvel[9],d->qvel[10]};
}

output sim(std::array<double,5>actions) {
      static int call{0};

      std::copy(actions.begin(),actions.end(),d->ctrl);
      mj_step(m, d);
    
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      // update scene and render
      mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
      mjr_render(viewport, &scn, &con);

   

       //swap OpenGL buffers (blocking call due to v-sync)
      glfwSwapBuffers(window);

      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();
      mjtNum* qpos = d->qpos;

  
      float reward=1;//-((std::abs(glm::radians(d->qpos[8]))+std::abs(glm::radians(d->qpos[10])))/2)-(std::abs(d->qpos[4])+std::abs(d->qpos[3])+std::abs(d->qpos[5]))/2;

      bool terminated = false;
      bool truncated = false;

      for(int i{0};i<d->ncon;i++){
        mjContact contact = d->contact[i];
        if(contact.geom[0]==0||contact.geom[1]==0){
          reward = -100;
          terminated = true;
          break;
        } 
      }
      glm::quat q{static_cast<float>(d->qpos[3]),static_cast<float>(d->qpos[4]),static_cast<float>(d->qpos[5]),static_cast<float>(d->qpos[6])};

      glm::vec3 angles = glm::degrees(glm::eulerAngles(q));

      reward = 1+(d->qvel[2]/5)-std::abs(angles.x/30);

      if(std::abs(angles.x)>70||std::abs(angles.y)>70||std::abs(angles.z)>70||d->qvel[2]<-5){
        reward = -100;
        terminated = true;
      }

      if((call+1)%100==0){
        mj_resetData(m,d);
      }
      call++;
      angles = glm::radians(angles);


      return{{d->qpos[2],angles.x,angles.y,angles.z,d->qpos[7],d->qpos[8],d->qpos[9],d->qpos[10],d->qpos[11],d->qvel[2],d->qvel[3],d->qvel[4],d->qvel[5],d->qvel[6],d->qvel[7],d->qvel[8],d->qvel[9],d->qvel[10]},reward,terminated,truncated};
}


PYBIND11_MODULE(rlenv, m) {
  py::class_<output>(m, "output")
    .def(py::init<>())
    .def_readwrite("state", &output::state)
    .def_readwrite("reward", &output::reward)
    .def_readwrite("terminated", &output::terminated)
    .def_readwrite("truncated", &output::truncated);
   
   
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("init",&sim_init,"");
    m.def("sim",&sim,"");
}

/*

qpos indices:   2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27

qvel indices: 


*/