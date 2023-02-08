import pyvista as pv
import numpy as np
from PIL import Image


class Render():
    def __init__(self, mesh, gif_name):
        self.normalize_to_bb(mesh)
        self.pyv_mesh = self.load_mesh(mesh)
        self.save_gif(gif_name)
    
    def load_mesh(self, mesh):
        vs = mesh.vs
        faces = np.ones([len(mesh.faces), 1]) * 3
        faces = np.concatenate([faces, mesh.faces], axis=1).astype(np.int32)
        pyv_mesh = pv.PolyData(vs, faces)
        return pyv_mesh
    
    def normalize_to_bb(self, mesh):
        vs = mesh.vs
        maxs = np.max(vs, axis=0, keepdims=True)
        mins = np.min(vs, axis=0, keepdims=True)
        ranges = maxs - mins
        vs = (vs - mins) / ranges
        mesh.vs = vs - 0.5
    
    def save_img(self, gif_name):
        plotter = pv.Plotter(off_screen=True, notebook=False)
        plotter.add_mesh(self.pyv_mesh, color="orange")
        plotter.show(screenshot=gif_name)
    
    def save_gif(self, gif_name):
        figures = []
        plotter = pv.Plotter(off_screen=True, notebook=False, window_size=[400,300])
        plotter.set_focus([0,0,0])
        plotter.set_position([2,2,2])
        for i in range(60):
            rot = self.pyv_mesh.rotate_y(6 * i, inplace=False)
            plotter = pv.Plotter(off_screen=True, notebook=False, window_size=[400,300])
            plotter.set_focus([0,0,0])
            plotter.set_position([2,2,2])
            plotter.add_mesh(rot, color="orange")
            img = plotter.screenshot(filename=None, return_img=True)
            figures.append(Image.fromarray(img))
        figures[0].save(gif_name, save_all=True, append_images=figures[1:], optimize=True, duration=50, loop=0)