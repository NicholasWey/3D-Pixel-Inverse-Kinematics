#version 330 core

out vec4 fragColor;

uniform vec2 u_resolution;
uniform float u_time;

uniform vec3 u_cam_pos;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_fov_y;

uniform float u_floor_extent;
uniform float u_pixel_size;
uniform int u_max_steps;
uniform int u_shadow_steps;
uniform float u_far_distance;
uniform float u_hit_epsilon;
uniform float u_zoom_t;
uniform float u_exposure;
uniform float u_contrast;

uniform vec3 u_capsule_a0;
uniform vec3 u_capsule_b0;
uniform vec3 u_capsule_a1;
uniform vec3 u_capsule_b1;
uniform vec3 u_capsule_a2;
uniform vec3 u_capsule_b2;
uniform float u_capsule_r0;
uniform float u_capsule_r1;
uniform float u_capsule_r2;

uniform vec3 u_block_pos;
uniform vec3 u_block_half;
uniform vec3 u_target_pos;
uniform float u_target_radius;
uniform int u_show_prediction;
uniform vec3 u_predicted_landing;

const int MAX_STEPS_CAP = 320;
const int PALETTE_SIZE = 8;
const vec3 PALETTE[PALETTE_SIZE] = vec3[](
    vec3(0.080, 0.090, 0.100),
    vec3(0.160, 0.180, 0.200),
    vec3(0.270, 0.290, 0.320),
    vec3(0.400, 0.430, 0.470),
    vec3(0.560, 0.590, 0.640),
    vec3(0.730, 0.760, 0.800),
    vec3(0.860, 0.760, 0.500),
    vec3(0.930, 0.930, 0.920)
);

float hash12(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 nearest_palette(vec3 c) {
    vec3 best = PALETTE[0];
    float best_d = 1e9;
    for (int i = 0; i < PALETTE_SIZE; i++) {
        vec3 d = c - PALETTE[i];
        float dist = dot(d, d);
        if (dist < best_d) {
            best_d = dist;
            best = PALETTE[i];
        }
    }
    return best;
}

float sd_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sd_box(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sd_torus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xy) - t.x, p.z);
    return length(q) - t.y;
}

vec2 op_union(vec2 a, vec2 b) {
    return (a.x < b.x) ? a : b;
}

vec2 map_scene(vec3 p) {
    vec2 res = vec2(1e6, 0.0);

    float floor_sdf = sd_box(
        p - vec3(0.0, 0.0, -0.02),
        vec3(u_floor_extent, u_floor_extent, 0.02)
    );
    res = op_union(res, vec2(floor_sdf, 1.0));

    res = op_union(res, vec2(sd_capsule(p, u_capsule_a0, u_capsule_b0, u_capsule_r0), 2.0));
    res = op_union(res, vec2(sd_capsule(p, u_capsule_a1, u_capsule_b1, u_capsule_r1), 2.0));
    res = op_union(res, vec2(sd_capsule(p, u_capsule_a2, u_capsule_b2, u_capsule_r2), 2.0));

    float block_sdf = sd_box(p - u_block_pos, u_block_half);
    res = op_union(res, vec2(block_sdf, 3.0));

    vec3 target_local = p - vec3(u_target_pos.xy, 0.020);
    float target_ring = sd_torus(target_local, vec2(u_target_radius, 0.017));
    res = op_union(res, vec2(target_ring, 4.0));

    if (u_show_prediction == 1) {
        vec3 pred_local = p - vec3(u_predicted_landing.xy, 0.018);
        float pred_ring = sd_torus(pred_local, vec2(0.06, 0.010));
        res = op_union(res, vec2(pred_ring, 5.0));
    }

    return res;
}

vec3 calc_normal(vec3 p) {
    float e = max(0.0009, u_hit_epsilon * 0.9);
    vec2 h = vec2(e, 0.0);
    float dx = map_scene(p + vec3(h.x, h.y, h.y)).x - map_scene(p - vec3(h.x, h.y, h.y)).x;
    float dy = map_scene(p + vec3(h.y, h.x, h.y)).x - map_scene(p - vec3(h.y, h.x, h.y)).x;
    float dz = map_scene(p + vec3(h.y, h.y, h.x)).x - map_scene(p - vec3(h.y, h.y, h.x)).x;
    return normalize(vec3(dx, dy, dz));
}

float soft_shadow(vec3 ro, vec3 rd) {
    float res = 1.0;
    float t = 0.012;
    float t_limit = min(14.0, u_far_distance * 0.45);
    for (int i = 0; i < 72; i++) {
        if (i >= u_shadow_steps) {
            break;
        }
        vec2 h = map_scene(ro + rd * t);
        if (h.x < 0.001) {
            return 0.0;
        }
        res = min(res, 8.0 * h.x / t);
        t += clamp(h.x, 0.01, 0.20);
        if (t > t_limit) {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}

vec3 sky_color(vec3 rd) {
    float horizon = clamp(0.5 + 0.5 * rd.z, 0.0, 1.0);
    vec3 low = vec3(0.30, 0.46, 0.70);
    vec3 high = vec3(0.58, 0.72, 0.90);
    vec3 c = mix(low, high, pow(horizon, 0.65));
    float cloud = 0.02 * sin(rd.x * 8.5 + u_time * 0.10) * sin(rd.y * 7.0 - u_time * 0.08);
    return c + cloud;
}

vec3 floor_color(vec3 p) {
    vec2 cell = floor(p.xy * 3.2);
    float checker = mod(cell.x + cell.y, 2.0);
    vec3 a = vec3(0.24, 0.27, 0.31);
    vec3 b = vec3(0.18, 0.21, 0.25);
    float noise = hash12(cell * 0.31) * 0.04;
    return mix(a, b, checker) + noise;
}

vec3 shade_hit(vec3 ro, vec3 rd, vec3 p, vec3 n, float mat_id) {
    vec3 key_dir = normalize(vec3(-0.54, 0.62, 0.84));
    vec3 fill_dir = normalize(vec3(0.42, -0.30, 0.66));
    float key = max(dot(n, key_dir), 0.0);
    float key_shadow = soft_shadow(p + n * 0.004, key_dir);
    float key_lit = floor((key * key_shadow) * 5.0 + 0.5) / 5.0;
    float fill = max(dot(n, fill_dir), 0.0);
    float hemi = mix(0.32, 0.98, clamp(n.z * 0.5 + 0.5, 0.0, 1.0));
    vec3 half_vec = normalize(key_dir - rd);
    float spec = pow(max(dot(n, half_vec), 0.0), 44.0) * 0.22 * key_shadow;

    vec3 base = vec3(0.4);
    vec3 emissive = vec3(0.0);
    if (mat_id < 1.5) {
        base = floor_color(p);
    } else if (mat_id < 2.5) {
        base = vec3(0.64, 0.68, 0.73);
    } else if (mat_id < 3.5) {
        base = vec3(0.96, 0.44, 0.23);
    } else if (mat_id < 4.5) {
        base = vec3(0.26, 0.76, 0.98);
        emissive = vec3(0.09, 0.20, 0.25);
    } else {
        base = vec3(0.98, 0.67, 0.20);
        emissive = vec3(0.23, 0.14, 0.05);
    }

    vec3 color = base * (0.16 + 0.74 * key_lit + 0.24 * fill + 0.20 * hemi) + spec + emissive;
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 2.6);
    color += base * rim * mix(0.14, 0.30, u_zoom_t);
    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 2.4);
    color *= 1.0 + 0.08 * fresnel;
    return color;
}

vec3 trace_scene(vec3 ro, vec3 rd) {
    float t = 0.0;
    float mat_id = -1.0;
    bool hit = false;

    for (int i = 0; i < MAX_STEPS_CAP; i++) {
        if (i >= u_max_steps) {
            break;
        }
        vec3 p = ro + rd * t;
        vec2 scene = map_scene(p);
        if (scene.x < u_hit_epsilon) {
            mat_id = scene.y;
            hit = true;
            break;
        }
        t += clamp(scene.x * 0.72, 0.0008, 0.42);
        if (t > u_far_distance) {
            break;
        }
    }

    if (!hit) {
        return sky_color(rd);
    }

    vec3 p = ro + rd * t;
    vec3 n = calc_normal(p);
    vec3 col = shade_hit(ro, rd, p, n, mat_id);
    float fog_density = mix(0.030, 0.014, u_zoom_t);
    float fog = exp(-fog_density * t);
    return mix(sky_color(rd), col, fog);
}

void main() {
    vec2 frag = gl_FragCoord.xy;
    vec2 sample_coord = frag;
    if (u_pixel_size > 1.0) {
        sample_coord = (floor(frag / u_pixel_size) + 0.5) * u_pixel_size;
    }
    vec2 uv = (sample_coord * 2.0 - u_resolution) / u_resolution.y;
    float tan_half_fov = tan(0.5 * u_fov_y);
    vec3 rd = normalize(
        u_cam_forward
        + uv.x * tan_half_fov * u_cam_right
        + uv.y * tan_half_fov * u_cam_up
    );

    vec3 color = trace_scene(u_cam_pos, rd);
    color = vec3(1.0) - exp(-color * u_exposure);
    color = pow(color, vec3(0.93));
    color = (color - 0.5) * u_contrast + 0.5;
    float dith = (hash12(floor(frag)) - 0.5) / 110.0;
    color = clamp(color + dith, 0.0, 1.0);
    vec3 palette_color = nearest_palette(color);
    float palette_strength = mix(1.0, 0.72, u_zoom_t);
    color = mix(color, palette_color, palette_strength);
    fragColor = vec4(color, 1.0);
}
