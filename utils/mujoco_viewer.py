import mujoco
import mujoco.viewer
import glfw




class BaseViewer:
    def __init__(self, model_path, distance=3, azimuth=0, elevation=-30):
        print(f"model_path : {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = distance
        self.handle.cam.azimuth = azimuth
        self.handle.cam.elevation = elevation
        self.opt = mujoco.MjvOption()

        # ✅ 正确使用 glfw（模块调用，不加 self.）
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(640, 480, "offscreen", None, None)
        glfw.make_context_current(self.window)

        # mujoco 相机数据相关
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.cur_episode_done = False


    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewportW

    def run_loop(self):
        self.run_before()
        while self.is_running():
            self.run_func()
            mujoco.mj_step(self.model, self.data)
            self.sync()
            if self.cur_episode_done == True:
                break
    
    def run_before(self):
        pass
        

    def run_func(self):
        pass