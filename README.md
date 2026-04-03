如果出現

=========================================================================

QObject::moveToThread: Current thread (0x5602786dac40) is not the object's thread (0x5602816923c0).
Cannot move to target thread (0x5602786dac40)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/ncku_csie_rl/miniconda3/envs/Test_Env/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.

=========================================================================

1. 徹底閹割 OpenCV 的 Qt 插件
rm -rf /home/ncku_csie_rl/miniconda3/envs/Test_Env/lib/python3.10/site-packages/cv2/qt/plugins

2. 強制讓系統只看 PyQt5 的插件路徑
export QT_QPA_PLATFORM_PLUGIN_PATH=/home/ncku_csie_rl/miniconda3/envs/Test_Env/lib/python3.10/site-packages/PyQt5/Qt5/plugins
export DISPLAY=:0
xhost +local:root > /dev/null 2>&1

"/home/ncku_csie_rl/miniconda3/envs/Test_Env/" 根據自己的路徑改變

=========================================================================

如果改完出現

=========================================================================

qt.qpa.xcb: could not connect to display :0
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/ncku_csie_rl/miniconda3/envs/Test_Env/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.

=========================================================================

則執行
export DISPLAY=:1



