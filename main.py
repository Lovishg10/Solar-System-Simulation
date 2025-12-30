import pygame
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3
import random
import math
import os
import sys
import traceback

# --- 0. DIRECTORY SETUP ---
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Working Directory: {os.getcwd()}")

# --- 1. SETTINGS ---
PARTICLE_COUNT = 200000
RING_PARTICLE_COUNT = 80000
G = 0.5
BASE_DT = 0.01
MOUSE_SENSITIVITY = 0.1
MAX_SPEED = 64.0
Z_FAR = 10000000.0

# --- 2. HELPERS ---
def safe_uniform(prog, name, value):
    if name not in prog: return
    try:
        if isinstance(value, (int, float)):
            prog[name].value = value
        elif isinstance(value, (tuple, list, np.ndarray)):
            val_np = np.array(value, dtype='f4')
            if val_np.ndim == 1 and val_np.size <= 4:
                prog[name].value = tuple(val_np)
            else:
                prog[name].write(val_np.tobytes())
    except Exception: pass

# --- 3. SHADERS ---
PLANET_VS = """
#version 330
in vec3 in_position; in vec3 in_normal; in vec2 in_uv;
uniform mat4 m_proj; uniform mat4 m_view; uniform mat4 m_model;
out vec3 v_normal; out vec3 v_pos; out vec2 v_uv;
void main() {
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
    v_normal = mat3(m_model) * in_normal;
    v_pos = vec3(m_model * vec4(in_position, 1.0));
    v_uv = in_uv;
}
"""
PLANET_FS = """
#version 330
in vec3 v_normal; in vec3 v_pos; in vec2 v_uv;
uniform sampler2D tex; uniform vec3 light_pos; uniform bool is_sun; uniform vec3 cam_pos;
out vec4 f_color;
void main() {
    vec4 texColor = texture(tex, v_uv);
    if (is_sun) { f_color = texColor; } 
    else {
        vec3 norm = normalize(v_normal);
        vec3 lightDir = normalize(light_pos - v_pos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * texColor.rgb;
        vec3 viewDir = normalize(cam_pos - v_pos);
        float fresnel = pow(1.0 - max(dot(norm, viewDir), 0.0), 3.0);
        vec3 atmosphere = vec3(0.3, 0.4, 0.6) * fresnel * diff;
        f_color = vec4(0.1 * texColor.rgb + diffuse + atmosphere, 1.0);
    }
}
"""

GALAXY_VS = """
#version 330
in vec3 in_position; in vec2 in_uv;
uniform mat4 m_proj; uniform mat4 m_view; uniform mat4 m_model;
uniform vec3 cam_right; uniform vec3 cam_up; uniform float scale;
out vec2 v_uv;
void main() {
    vec3 center = vec3(m_model[3][0], m_model[3][1], m_model[3][2]);
    vec3 pos = center + (cam_right * in_position.x * scale) + (cam_up * in_position.y * scale);
    gl_Position = m_proj * m_view * vec4(pos, 1.0);
    v_uv = in_uv;
}
"""
GALAXY_FS = """
#version 330
in vec2 v_uv; uniform sampler2D tex; uniform vec3 tint; out vec4 f_color;
void main() {
    vec4 color = texture(tex, v_uv);
    float brightness = dot(color.rgb, vec3(0.3, 0.59, 0.11));
    float alpha = smoothstep(0.1, 0.4, brightness);
    vec3 high_contrast = pow(color.rgb, vec3(1.5));
    f_color = vec4(high_contrast * tint * 2.0 * alpha, alpha);
}
"""

GLOW_VS = """
#version 330
in vec3 in_position; in vec2 in_uv; 
uniform mat4 m_proj; uniform mat4 m_view; uniform mat4 m_model; 
uniform vec3 cam_right; uniform vec3 cam_up; 
out vec2 v_uv;
void main() {
    vec3 center = vec3(m_model[3][0], m_model[3][1], m_model[3][2]);
    vec3 pos = center + (cam_right * in_position.x * 30.0) + (cam_up * in_position.y * 30.0);
    gl_Position = m_proj * m_view * vec4(pos, 1.0);
    v_uv = in_uv; 
}
"""
GLOW_FS = """
#version 330
in vec2 v_uv; uniform vec3 color; out vec4 f_color; 
void main() { 
    float dist = distance(v_uv, vec2(0.5)); 
    float alpha = 1.0 - smoothstep(0.0, 0.5, dist); 
    alpha = pow(alpha, 3.0);
    f_color = vec4(color, alpha * 0.8); 
}
"""

PARTICLE_VS = "#version 330\nin vec3 in_pos; in vec3 in_vel; uniform mat4 m_proj; uniform mat4 m_view; out float v_speed;\nvoid main() { gl_Position = m_proj * m_view * vec4(in_pos, 1.0); gl_PointSize = 1.0; v_speed = length(in_vel); }"
PARTICLE_FS = "#version 330\nin float v_speed; out vec4 f_color; void main() { vec3 col = mix(vec3(0.2, 0.2, 0.3), vec3(0.8, 0.8, 0.9), min(v_speed * 0.3, 1.0)); f_color = vec4(col, 0.6); }"

PHYSICS_VS = "#version 330\nin vec3 in_pos; in vec3 in_vel; out vec3 out_pos; out vec3 out_vel; uniform float dt; uniform vec3 sun_pos;\nvoid main() { vec3 pos = in_pos; vec3 vel = in_vel; vec3 diff = sun_pos - pos; float dist_sq = dot(diff, diff);\nvec3 dir = diff / sqrt(max(dist_sq, 0.001)); float force = 5000.0 / max(dist_sq, 100.0);\nvel += dir * force * dt; pos += vel * dt; if (dist_sq > 25000000.0 || dist_sq < 10000.0) pos = -pos; \nout_pos = pos; out_vel = vel; }"

SIMPLE_VS = "#version 330\nin vec3 in_position; in vec3 in_color; uniform mat4 m_proj; uniform mat4 m_view; uniform mat4 m_model; out vec3 v_color;\nvoid main() { gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0); v_color = in_color; }"
SIMPLE_FS = "#version 330\nin vec3 v_color; out vec4 f_color; void main() { f_color = vec4(v_color, 1.0); }"

LINE_VS = "#version 330\nin vec3 in_position; uniform mat4 m_proj; uniform mat4 m_view; uniform mat4 m_model; void main() { gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0); }"
LINE_FS = "#version 330\nuniform vec3 color; out vec4 f_color; void main() { f_color = vec4(color, 1.0); }"

HUD_VS = "#version 330\nin vec2 in_position; in vec2 in_uv; out vec2 v_uv; void main() { gl_Position = vec4(in_position, 0.0, 1.0); v_uv = in_uv; }"
HUD_FS = "#version 330\nuniform sampler2D tex; in vec2 v_uv; out vec4 f_color; void main() { f_color = texture(tex, v_uv); }"

RING_PARTICLE_VS = "#version 330\nin vec3 in_pos; uniform mat4 m_proj; uniform mat4 m_view; uniform mat4 m_model;\nvoid main() { gl_Position = m_proj * m_view * m_model * vec4(in_pos, 1.0); gl_PointSize = 2.5; }"
RING_PARTICLE_FS = "#version 330\nout vec4 f_color; void main() { f_color = vec4(0.92, 0.85, 0.75, 0.85); }"

# --- 4. CLASSES ---
class TextureManager:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}
        self.galaxy_textures = []
        # Define the folder name here
        self.img_dir = "images" 
        
        # Create the folder if it doesn't exist (optional helper)
        if not os.path.exists(self.img_dir):
            print(f"Warning: '{self.img_dir}' folder not found. Textures may not load.")
            
        self.load_galaxies()

    def load_galaxies(self):
        # Check if directory exists to avoid crash
        if not os.path.exists(self.img_dir):
            self.galaxy_textures.append(self.generate_procedural("Galaxy"))
            return

        for i in range(1, 10):
            found = False
            for ext in [".jpg", ".png", ".jpeg"]:
                name = f"galaxy{i}{ext}"
                # Look inside the images directory
                for f in os.listdir(self.img_dir): 
                    if f.lower() == name:
                        try:
                            # Construct full path: images/galaxy1.jpg
                            full_path = os.path.join(self.img_dir, f)
                            img = pygame.image.load(full_path).convert_alpha()
                            img = pygame.transform.flip(img, False, True)
                            data = pygame.image.tobytes(img, 'RGBA', False)
                            tex = self.ctx.texture(img.get_size(), 4, data)
                            tex.build_mipmaps()
                            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                            tex.anisotropy = 16.0
                            self.galaxy_textures.append(tex)
                            found = True
                        except: pass
                        break
                if found: break
        if not self.galaxy_textures:
            self.galaxy_textures.append(self.generate_procedural("Galaxy"))

    def get_galaxy_texture(self, index):
        return self.galaxy_textures[index % len(self.galaxy_textures)]

    def get_texture(self, name):
        if name in self.textures: return self.textures[name]
        
        for n in [name, name.lower(), name.upper(), name.capitalize()]:
            for ext in [".jpg", ".png", ".jpeg"]:
                filename = n + ext
                # Construct the full path to check: images/Earth.jpg
                full_path = os.path.join(self.img_dir, filename)
                
                if os.path.exists(full_path):
                    try:
                        img = pygame.image.load(full_path).convert_alpha()
                        img = pygame.transform.flip(img, False, True)
                        data = pygame.image.tobytes(img, 'RGBA', False)
                        tex = self.ctx.texture(img.get_size(), 4, data)
                        tex.build_mipmaps()
                        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                        tex.anisotropy = 16.0
                        self.textures[name] = tex
                        return tex
                    except Exception as e:
                        print(f"Failed to load {full_path}: {e}")

        # If we reach here, no file was found in the images folder
        return self.generate_procedural(name)

    def generate_procedural(self, name):
        size = 256
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        if "Galaxy" in name:
            for r in range(100, 0, -5):
                a = int(255 * (1.0 - r/100.0))
                pygame.draw.circle(surf, (100, 120, 255, a), (128, 128), r)
            pygame.draw.circle(surf, (255, 255, 255, 255), (128, 128), 20)
        else:
            surf.fill((180, 180, 180, 255))
        data = pygame.image.tobytes(pygame.transform.flip(surf, False, True), 'RGBA', False)
        tex = self.ctx.texture((size, size), 4, data)
        tex.build_mipmaps()
        return tex

class Body3D:
    def __init__(self, name, mass, pos, vel, color, scale, trail_length, orbital_radius=0):
        self.name = name; self.mass = mass; self.pos = np.array(pos, dtype='f4'); self.vel = np.array(vel, dtype='f4')
        self.color = color; self.scale = scale; self.trail = []; self.max_trail = trail_length; self.orbital_radius = orbital_radius
        self.rot_speed = 1.0

class Moon3D:
    def __init__(self, name, parent, distance, speed, color, scale):
        self.name = name; self.parent = parent; self.distance = distance; self.angle = random.uniform(0, 6.28); self.speed = speed
        self.color = color; self.scale = scale; self.pos = np.array([0.0, 0.0, 0.0], dtype='f4')

class Galaxy3D:
    def __init__(self, pos, scale, tint, tex_index):
        self.pos = pos; self.scale = scale; self.tint = tint; self.tex_index = tex_index

# --- 5. GENERATORS ---
def generate_sphere(radius=1.0, stacks=32, slices=32):
    vertices = []; indices = []
    for i in range(stacks + 1):
        lat = np.pi * i / stacks; sin_lat = np.sin(lat); cos_lat = np.cos(lat)
        for j in range(slices + 1):
            lon = 2 * np.pi * j / slices; x = np.cos(lon) * sin_lat; y = cos_lat; z = np.sin(lon) * sin_lat; u = 1 - (j / slices); v = 1 - (i / stacks)
            vertices.extend([x*radius, y*radius, z*radius, x, y, z, u, v])
    for i in range(stacks):
        for j in range(slices):
            first = (i * (slices + 1)) + j; second = first + slices + 1
            indices.extend([first, second, first + 1, second, second + 1, first + 1])
    return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')

def generate_quad():
    return np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype='f4')

def generate_gpu_particles(count):
    data = np.zeros(count * 6, dtype='f4')
    angles = np.random.uniform(0, 2*np.pi, count); radii = np.random.uniform(600, 850, count); heights = np.random.normal(0, 15, count)
    data[0::6] = radii * np.cos(angles); data[1::6] = heights; data[2::6] = radii * np.sin(angles)
    vel_mags = np.sqrt((0.5 * 10000.0) / radii)
    data[3::6] = -np.sin(angles) * vel_mags; data[4::6] = 0.0; data[5::6] = np.cos(angles) * vel_mags
    return data

def generate_orbit_path(radius, points=360):
    data = []
    for i in range(points):
        angle = (i / points) * 2 * math.pi
        data.extend([math.cos(angle) * radius, 0.0, math.sin(angle) * radius])
    return np.array(data, dtype='f4')

def generate_background_stars(count=10000):
    data = []
    for _ in range(count):
        x = random.uniform(-100000, 100000); y = random.uniform(-100000, 100000); z = random.uniform(-100000, 100000)
        c = random.uniform(0.4, 0.9); data.extend([x, y, z, c, c, c])
    return data

def generate_saturn_ring_particles(count):
    data = np.zeros(count * 3, dtype='f4')
    angles = np.random.uniform(0, 2*np.pi, count)
    
    # [FIX] RELATIVE UNITS: 
    # Use relative units (1.2 to 2.3) so when multiplied by planet scale (7.0),
    # they create a tight, realistic ring system.
    radii = np.random.uniform(1.2, 2.3, count) 
    
    # [FIX] THINNER RINGS: Reduced std deviation for height
    heights = np.random.normal(0, 0.05, count)
    
    data[0::3] = radii * np.cos(angles)
    data[1::3] = heights
    data[2::3] = radii * np.sin(angles)
    return data

def create_text_texture(ctx, text):
    font = pygame.font.SysFont("arial", 20, bold=True)
    text_surf = font.render(text, True, (255, 255, 255))
    surf = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    text_surf.set_alpha(128)
    surf.blit(text_surf, (0, 0))
    data = pygame.image.tobytes(pygame.transform.flip(surf, False, True), 'RGBA', False)
    texture = ctx.texture(surf.get_size(), 4, data)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return texture

def create_solar_system():
    sun_mass = 10000.0
    sun = Body3D("Sun", sun_mass, [0,0,0], [0,0,0], (1.0, 0.8, 0.0), 12.0, 0)
    bodies = [sun]; moons = []
    def add_planet(name, dist, mass, color, scale, trail):
        angle = random.uniform(0, 6.28); px = math.cos(angle) * dist; pz = math.sin(angle) * dist
        vel = math.sqrt((G * sun_mass) / dist)
        vx = -math.sin(angle) * vel; vz = math.cos(angle) * vel
        new_body = Body3D(name, mass, [px, 0, pz], [vx, 0, vz], color, scale, trail, dist)
        bodies.append(new_body); return new_body

    add_planet("Mercury", 200, 0.05, (0.7,0.7,0.7), 1.5, 30)
    add_planet("Venus", 300, 0.4, (0.9,0.8,0.5), 2.8, 40)
    earth = add_planet("Earth", 420, 0.5, (0.2,0.5,1.0), 3.0, 50)
    add_planet("Mars", 550, 0.3, (1.0,0.3,0.2), 2.2, 60)
    jup = add_planet("Jupiter", 900, 10.0, (0.8,0.7,0.5), 8.0, 300)
    sat = add_planet("Saturn", 1300, 8.0, (0.9,0.8,0.6), 7.0, 400)
    add_planet("Uranus", 1700, 4.0, (0.5,0.8,0.9), 5.0, 500)
    add_planet("Neptune", 2100, 4.0, (0.3,0.3,0.8), 5.0, 600)
    add_planet("Pluto", 2500, 0.02, (0.6,0.5,0.5), 1.0, 800)

    moons.append(Moon3D("Moon", earth, 15, 0.05, (0.8,0.8,0.8), 0.8))
    moons.append(Moon3D("Io", jup, 18, 0.08, (1.0,1.0,0.0), 0.9))
    moons.append(Moon3D("Europa", jup, 24, 0.06, (0.9,0.9,1.0), 0.8))
    moons.append(Moon3D("Ganymede", jup, 32, 0.04, (0.7,0.7,0.7), 1.2))
    moons.append(Moon3D("Callisto", jup, 40, 0.03, (0.5,0.5,0.5), 1.1))
    
    # [FIX] TITAN DISTANCE:
    # Saturn Scale is 7.0. Rings now end at approx 16.5 relative units.
    # Set Titan distance to 35 to ensure it is clearly outside the rings.
    moons.append(Moon3D("Titan", sat, 35, 0.04, (0.9,0.6,0.2), 1.2))
    
    return bodies, moons, sat

# --- 7. MAIN ---
def main():
    try:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        width, height = 1280, 720
        window = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.mouse.set_visible(False); pygame.event.set_grab(True)
        pygame.mouse.set_pos(width//2, height//2)
        ctx = moderngl.create_context()
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        prog_planet = ctx.program(vertex_shader=PLANET_VS, fragment_shader=PLANET_FS)
        prog_phys = ctx.program(vertex_shader=PHYSICS_VS, varyings=['out_pos', 'out_vel'])
        prog_part = ctx.program(vertex_shader=PARTICLE_VS, fragment_shader=PARTICLE_FS)
        prog_simple = ctx.program(vertex_shader=SIMPLE_VS, fragment_shader=SIMPLE_FS)
        prog_line = ctx.program(vertex_shader=LINE_VS, fragment_shader=LINE_FS)
        prog_hud = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        prog_glow = ctx.program(vertex_shader=GLOW_VS, fragment_shader=GLOW_FS)
        prog_galaxy = ctx.program(vertex_shader=GALAXY_VS, fragment_shader=GALAXY_FS)
        prog_ring_part = ctx.program(vertex_shader=RING_PARTICLE_VS, fragment_shader=RING_PARTICLE_FS)

        tex_man = TextureManager(ctx)

        initial_data = generate_gpu_particles(PARTICLE_COUNT)
        vbo1 = ctx.buffer(initial_data); vao1 = ctx.vertex_array(prog_part, [(vbo1, '3f 3f', 'in_pos', 'in_vel')])
        t_vao1 = ctx.vertex_array(prog_phys, [(vbo1, '3f 3f', 'in_pos', 'in_vel')])
        vbo2 = ctx.buffer(reserve=initial_data.nbytes)
        vao2 = ctx.vertex_array(prog_part, [(vbo2, '3f 3f', 'in_pos', 'in_vel')])
        t_vao2 = ctx.vertex_array(prog_phys, [(vbo2, '3f 3f', 'in_pos', 'in_vel')])

        sphere_vbo, sphere_ibo = generate_sphere(1.0)
        vbo_sphere = ctx.buffer(sphere_vbo); ibo_sphere = ctx.buffer(sphere_ibo)
        vao_sphere = ctx.vertex_array(prog_planet, [(vbo_sphere, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')], ibo_sphere)

        uni_data = generate_background_stars(10000)
        vbo_uni = ctx.buffer(np.array(uni_data, dtype='f4'))
        vao_uni = ctx.vertex_array(prog_simple, [(vbo_uni, '3f 3f', 'in_position', 'in_color')])

        glow_data = generate_quad()
        vbo_glow = ctx.buffer(glow_data)
        vao_glow = ctx.vertex_array(prog_glow, [(vbo_glow, '3f 2f', 'in_position', 'in_uv')])
        vbo_galaxy = ctx.buffer(glow_data)
        vao_galaxy = ctx.vertex_array(prog_galaxy, [(vbo_galaxy, '3f 2f', 'in_position', 'in_uv')])

        vbo_trail = ctx.buffer(reserve=20000 * 12)
        vao_trail = ctx.vertex_array(prog_line, [(vbo_trail, '3f', 'in_position')])

        bodies, moons, saturn_body = create_solar_system()
        for b in bodies:
            b.rot_speed = random.uniform(0.5, 2.5)
            if b.name == "Sun": b.rot_speed = 0.2
            if b.name == "Venus": b.rot_speed = 0.05

        orbit_vaos = []
        for b in bodies:
            if b.orbital_radius > 0:
                pts = generate_orbit_path(b.orbital_radius)
                vbo = ctx.buffer(pts)
                vao = ctx.vertex_array(prog_line, [(vbo, '3f', 'in_position')])
                orbit_vaos.append(vao)

        hud_data = np.array([0.7, -0.95, 0.0, 0.0, 0.95, -0.95, 1.0, 0.0, 0.7, -0.85, 0.0, 1.0, 0.7, -0.85, 0.0, 1.0, 0.95, -0.95, 1.0, 0.0, 0.95, -0.85, 1.0, 1.0], dtype='f4')
        vbo_hud = ctx.buffer(hud_data)
        vao_hud = ctx.vertex_array(prog_hud, [(vbo_hud, '2f 2f', 'in_position', 'in_uv')])

        ring_data = generate_saturn_ring_particles(RING_PARTICLE_COUNT)
        vbo_ring = ctx.buffer(ring_data)
        vao_ring = ctx.vertex_array(prog_ring_part, [(vbo_ring, '3f', 'in_pos')])

        galaxies = []
        for i in range(15):
            phi = math.acos(1 - 2 * (i + 0.5) / 15)
            theta = math.pi * (1 + 5**0.5) * (i + 0.5)
            x, y, z = math.cos(theta)*math.sin(phi), math.cos(phi), math.sin(theta)*math.sin(phi)
            dist = random.uniform(2000000, 3000000)
            pos = Vector3([x, y, z]) * dist
            scale = random.uniform(300000, 500000)
            tint = (random.uniform(0.7,1.0), random.uniform(0.7,1.0), random.uniform(0.7,1.0))
            galaxies.append(Galaxy3D(pos, scale, tint, i))

        camera_pos = Vector3([0.0, 800.0, 1500.0])
        yaw, pitch = -90.0, -30.0
        clock = pygame.time.Clock()
        speed_mult = 1.0; running = True; paused = False; swap = True; time = 0.0
        hud_tex = create_text_texture(ctx, f"Speed: {speed_mult:.1f}x")

        while running:
            dt = clock.tick(60) / 1000.0
            time += dt
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False
                    if event.key == pygame.K_l: speed_mult = min(speed_mult * 2.0, MAX_SPEED); hud_tex = create_text_texture(ctx, "PAUSED" if paused else f"Speed: {speed_mult:.1f}x")
                    if event.key == pygame.K_k: speed_mult = max(speed_mult / 2.0, 1.0); hud_tex = create_text_texture(ctx, "PAUSED" if paused else f"Speed: {speed_mult:.1f}x")
                    if event.key == pygame.K_p: paused = not paused; hud_tex = create_text_texture(ctx, "PAUSED" if paused else f"Speed: {speed_mult:.1f}x")
                if event.type == pygame.VIDEORESIZE: width, height = event.w, event.h; ctx.viewport = (0, 0, width, height)

            mx, my = pygame.mouse.get_pos()
            yaw += (mx - width//2) * MOUSE_SENSITIVITY
            pitch -= (my - height//2) * MOUSE_SENSITIVITY
            pitch = max(-89.0, min(89.0, pitch))
            pygame.mouse.set_pos(width//2, height//2)
            front = Vector3([math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)), math.sin(math.radians(pitch)), math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))])
            cam_front = front.normalized; cam_up = Vector3([0.0,1.0,0.0])
            cam_right = Vector3(np.cross(cam_front, cam_up)).normalized

            keys = pygame.key.get_pressed(); s = 50 * dt
            if keys[pygame.K_LSHIFT]: s *= 70.0
            if keys[pygame.K_w]: camera_pos += cam_front * s
            if keys[pygame.K_s]: camera_pos -= cam_front * s
            if keys[pygame.K_a]: camera_pos -= cam_right * s
            if keys[pygame.K_d]: camera_pos += cam_right * s
            if keys[pygame.K_SPACE]: camera_pos.y += s
            if keys[pygame.K_LCTRL]: camera_pos.y -= s

            if not paused:
                safe_uniform(prog_phys, 'dt', BASE_DT * speed_mult)
                safe_uniform(prog_phys, 'sun_pos', bodies[0].pos)
                if swap: t_vao1.transform(vbo2, mode=moderngl.POINTS)
                else: t_vao2.transform(vbo1, mode=moderngl.POINTS)
                loops = int(10 * speed_mult); loops = min(loops, 100)
                for _ in range(loops):
                    for b in bodies:
                        if b.name == "Sun": continue
                        diff = bodies[0].pos - b.pos; dist = np.linalg.norm(diff)
                        f = (G * bodies[0].mass) / (dist**2)
                        b.vel += (diff/dist) * f * BASE_DT
                        b.pos += b.vel * BASE_DT
                for b in bodies:
                    if b.name == "Sun": continue
                    if len(b.trail) == 0 or np.linalg.norm(b.pos - b.trail[-1]) > 4.0:
                        b.trail.append(b.pos.copy())
                    if len(b.trail) > b.max_trail: b.trail.pop(0)
                moon_dt = BASE_DT * loops
                for m in moons:
                    m.angle += m.speed * moon_dt * 2.0
                    px = m.parent.pos[0] + math.cos(m.angle) * m.distance
                    pz = m.parent.pos[2] + math.sin(m.angle) * m.distance
                    m.pos = np.array([px, 0.0, pz])

            ctx.clear(0.0, 0.0, 0.0)
            m_proj = Matrix44.perspective_projection(45.0, width/height, 0.1, Z_FAR)
            m_view = Matrix44.look_at(camera_pos, camera_pos + cam_front, cam_up)
            m_view_skybox = m_view.copy(); m_view_skybox[3][0] = m_view_skybox[3][1] = m_view_skybox[3][2] = 0.0

            # Galaxies
            ctx.disable(moderngl.DEPTH_TEST); ctx.depth_mask = False
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            safe_uniform(prog_galaxy, 'm_proj', m_proj); safe_uniform(prog_galaxy, 'm_view', m_view_skybox)
            safe_uniform(prog_galaxy, 'cam_right', cam_right); safe_uniform(prog_galaxy, 'cam_up', cam_up)
            for g in galaxies:
                m_model = Matrix44.from_translation(g.pos)
                safe_uniform(prog_galaxy, 'm_model', m_model)
                safe_uniform(prog_galaxy, 'scale', g.scale); safe_uniform(prog_galaxy, 'tint', g.tint)
                tex_man.get_galaxy_texture(g.tex_index).use(0)
                vao_galaxy.render(mode=moderngl.TRIANGLE_STRIP)
            ctx.enable(moderngl.DEPTH_TEST); ctx.depth_mask = True
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            # Orbits, stars, trails, disk
            safe_uniform(prog_line, 'm_proj', m_proj); safe_uniform(prog_line, 'm_view', m_view); safe_uniform(prog_line, 'm_model', Matrix44.identity())
            safe_uniform(prog_line, 'color', (0.3, 0.3, 0.3))
            for vao in orbit_vaos: vao.render(mode=moderngl.LINE_LOOP)
            safe_uniform(prog_simple, 'm_proj', m_proj); safe_uniform(prog_simple, 'm_view', m_view); safe_uniform(prog_simple, 'm_model', Matrix44.identity())
            vao_uni.render(mode=moderngl.POINTS)
            for b in bodies:
                if len(b.trail) < 2: continue
                trail_data = np.array(b.trail, dtype='f4').flatten()
                vbo_trail.write(trail_data.tobytes())
                safe_uniform(prog_line, 'color', b.color)
                vao_trail.render(mode=moderngl.LINE_STRIP, vertices=len(b.trail))
            safe_uniform(prog_part, 'm_proj', m_proj); safe_uniform(prog_part, 'm_view', m_view)
            if swap: vao2.render(mode=moderngl.POINTS)
            else: vao1.render(mode=moderngl.POINTS)
            swap = not swap

            # Planets & Moons with rotation
            safe_uniform(prog_planet, 'm_proj', m_proj); safe_uniform(prog_planet, 'm_view', m_view)
            safe_uniform(prog_planet, 'light_pos', bodies[0].pos); safe_uniform(prog_planet, 'cam_pos', camera_pos)
            for b in bodies:
                rot = Matrix44.from_y_rotation(math.radians(time * b.rot_speed * 50))
                model = Matrix44.from_translation(b.pos) * rot * Matrix44.from_scale([b.scale]*3)
                safe_uniform(prog_planet, 'm_model', model)
                safe_uniform(prog_planet, 'is_sun', b.name == "Sun")
                tex_man.get_texture(b.name).use(0)
                vao_sphere.render()
            for m in moons:
                rot = Matrix44.from_y_rotation(math.radians(time * 100))
                model = Matrix44.from_translation(m.pos) * rot * Matrix44.from_scale([m.scale]*3)
                safe_uniform(prog_planet, 'm_model', model)
                safe_uniform(prog_planet, 'is_sun', False)
                tex_man.get_texture(m.name).use(0)
                vao_sphere.render()

            # [FIX] SATURN RINGS RENDER:
            # 1. Enable Depth Test so planet blocks ring.
            # 2. Disable Depth Mask to blend particles smoothly.
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.depth_mask = False 
            ctx.enable(moderngl.BLEND)
            
            ring_rot = Matrix44.from_x_rotation(math.radians(26.5)) * Matrix44.from_y_rotation(math.radians(time * 8))
            # The particles (radii 1.2-2.3) are multiplied here by saturn's scale (7.0)
            ring_model = (Matrix44.from_translation(saturn_body.pos) * ring_rot * Matrix44.from_scale([saturn_body.scale * 1.0, 1.0, saturn_body.scale * 1.0]))
            
            safe_uniform(prog_ring_part, 'm_proj', m_proj)
            safe_uniform(prog_ring_part, 'm_view', m_view)
            safe_uniform(prog_ring_part, 'm_model', ring_model)
            vao_ring.render(mode=moderngl.POINTS)
            
            ctx.depth_mask = True # Reset for next frame

            # Sun glow
            ctx.disable(moderngl.CULL_FACE)
            safe_uniform(prog_glow, 'm_proj', m_proj); safe_uniform(prog_glow, 'm_view', m_view)
            safe_uniform(prog_glow, 'm_model', Matrix44.from_translation(bodies[0].pos))
            safe_uniform(prog_glow, 'cam_right', cam_right); safe_uniform(prog_glow, 'cam_up', cam_up)
            safe_uniform(prog_glow, 'color', (1.0, 0.6, 0.2))
            vao_glow.render(mode=moderngl.TRIANGLE_STRIP)

            # HUD
            hud_tex.use(location=0); safe_uniform(prog_hud, 'tex', 0)
            vao_hud.render(mode=moderngl.TRIANGLES)
            ctx.enable(moderngl.CULL_FACE)

            pygame.display.flip()
        pygame.quit()
    except Exception as e:
        print(f"CRASH: {e}")
        traceback.print_exc()
        pygame.quit()

if __name__ == "__main__":
    main()