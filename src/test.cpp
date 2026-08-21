#include "mujoco/mujoco.h"
#include "glfw/glfw3.h"
#include "stdio.h"
#include <iostream>
#include <array>
#include <algorithm>


struct output{
  std::array<double,10>state;
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


std::array<double,10> sim_init(){
  m = mj_loadXML("hawk.xml", NULL, error, 1000);
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
  return {d->qpos[2],d->qpos[3],d->qpos[4],d->qpos[5],d->qpos[6],d->qpos[7],d->qpos[8],d->qvel[2],d->qvel[6],d->qvel[7]};
}





void sim() {
  for(int i{0};i<m->nbody;i++){
    std::cout<<m->body_mass[i]<<'\n';
  }
  int n{0};
  while(!glfwWindowShouldClose(window)){
    n++;
    mjtNum simstart = d->time;
    while( d->time - simstart< 1.0/60.0 ){
      d->ctrl[0] = -1;  // r_rot — tilt wings ~3° into wind
      d->ctrl[1] = -1;  // l_rot
      d->ctrl[2] = -1;  // t_joint
      d->ctrl[3] = -1.0; // r_flap — full downstroke
      d->ctrl[4] = -1.0;
        mj_step(m, d);
      }
 
      /*std::copy(actions.begin(),actions.end(),d->ctrl);
      mj_step(m, d);*/
    
      mjrRect viewport = {0, 0, 0, 0};
      glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

      // update scene and render
      mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
      mjr_render(viewport, &scn, &con);

   

      // swap OpenGL buffers (blocking call due to v-sync)
      glfwSwapBuffers(window);

      // process pending GUI events, call GLFW callbacks
      glfwPollEvents();
    

      for(int i{0};i<d->ncon;i++){
        mjContact contact = d->contact[i];
        if(contact.geom[0]==0&&(contact.geom[1]!=8&&contact.geom[1]!=9&&contact.geom[1]!=12&&contact.geom[1]!=13)||(contact.geom[0]!=8&&contact.geom[0]!=9&&contact.geom[0]!=12&&contact.geom[0]!=13)&&contact.geom[1]==0){
          mj_resetData(m,d);
          break;
        } 
      }
}}

int main(){
  sim_init();
  sim();
}

