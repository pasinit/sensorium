"""The offscreen-rendering backend chosen by ``sensorium.bodies`` before mujoco is imported."""

from sensorium.bodies import gl_defaults


def test_headless_linux_gets_egl_and_macos_glfw() -> None:
    assert gl_defaults("linux", {}) == {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    assert gl_defaults("darwin", {}) == {"MUJOCO_GL": "glfw"}


def test_a_display_or_an_explicit_choice_is_left_alone() -> None:
    assert gl_defaults("linux", {"DISPLAY": ":0"}) == {}
    assert gl_defaults("linux", {"WAYLAND_DISPLAY": "wayland-0"}) == {}
    assert gl_defaults("linux", {"MUJOCO_GL": "osmesa"}) == {}
    assert gl_defaults("darwin", {"MUJOCO_GL": "egl"}) == {"PYOPENGL_PLATFORM": "egl"}
    assert gl_defaults("linux", {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "osmesa"}) == {}
