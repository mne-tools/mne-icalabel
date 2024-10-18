#%%
import numpy as np

# Conversion functions
def cart2sph(x, y, z):
    xy = np.sqrt(x * x + y * y)
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, xy)
    return r, theta, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def make_head_outlines(sphere, pos, outlines, clip_origin):
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    ll = np.linspace(0, 2 * np.pi, 101)
    head_x = np.cos(ll) * radius * 1.01 + x
    head_y = np.sin(ll) * radius * 1.01 + y
    dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
    dx, dy = dx.real, dx.imag
    
    outlines_dict = dict(head=(head_x, head_y))
    
    mask_scale = 1.
    mask_scale = max(
        mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius
    )
    
    outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
    clip_radius = radius * mask_scale
    outlines_dict['clip_radius'] = (clip_radius,) * 2
    outlines_dict['clip_origin'] = clip_origin
    
    outlines = outlines_dict
    
    return outlines


