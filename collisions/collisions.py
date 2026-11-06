"""
collisions.py — Core Collision Detection Engine
=============================================
This module provides the core physics and mathematical calculations required
for detecting collisions and calculating Minimum Translation Vectors (MTV) 
between circles and squares. It relies heavily on the Separating Axis Theorem (SAT).
"""

import math

# ---------------------------------------------------------
# Mathematical Helpers for Distances and Coordinate Mapping
# ---------------------------------------------------------
def dist(x1, x2, y1, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2

def get_corners(square):
    half = square['r']            # half-side
    angle = square['orientation']
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = square['x'], square['y']
    local = [( half,  half), (-half,  half), (-half, -half), ( half, -half)]
    return [(cx + (lx * cos_a - ly * sin_a), cy + (lx * sin_a + ly * cos_a)) for lx, ly in local]

# ---------- MTV calculators (return (mtv_for_p1, mtv_for_p2) or None if no collision) ----------

def get_mtv_circle_circle(p1, p2, eps=1e-9):
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    d2 = dx*dx + dy*dy
    rsum = p1['r'] + p2['r']
    if d2 <= 0.0:
        # coincide: push along arbitrary axis
        nx, ny = 1.0, 0.0
        overlap = rsum
        mtv = (nx * overlap, ny * overlap)
        return mtv, (-mtv[0], -mtv[1])
    d = math.sqrt(d2)
    overlap = rsum - d
    if overlap <= eps:
        return None
    nx = dx / d
    ny = dy / d
    mtv = (nx * overlap, ny * overlap)
    return mtv, (-mtv[0], -mtv[1])

def get_mtv_circle_square(circle, square, eps=1e-9):
    # transform circle center into square local coordinates (unrotate)
    dx = circle['x'] - square['x']
    dy = circle['y'] - square['y']
    angle = -square['orientation']
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    local_x = dx * cos_a - dy * sin_a
    local_y = dx * sin_a + dy * cos_a

    half = square['r']
    # closest point on axis-aligned square in local coords
    clamped_x = max(-half, min(half, local_x))
    clamped_y = max(-half, min(half, local_y))

    diff_x = local_x - clamped_x
    diff_y = local_y - clamped_y
    d2 = diff_x * diff_x + diff_y * diff_y
    cr = circle['r']

    if d2 <= eps:
        # center is inside the square or exactly on an edge/corner:
        # compute minimal push direction to face with smallest distance
        # distances to faces in local coords
        dist_left = local_x + half
        dist_right = half - local_x
        dist_bottom = local_y + half
        dist_top = half - local_y
        # minimal positive distance to face (closest exit)
        face_dists = [(dist_right, (1, 0)), (dist_left, (-1, 0)), (dist_top, (0, 1)), (dist_bottom, (0, -1))]
        # select smallest positive distance
        face_dists = sorted(face_dists, key=lambda t: t[0])
        if face_dists[0][0] <= 0:
            # already overlapping strongly, pick a fallback
            local_nx, local_ny = 1.0, 0.0
            penetration = cr + abs(face_dists[0][0])
        else:
            local_nx, local_ny = face_dists[0][1]
            # penetration = circle_radius + distance from center to face (we must push circle out)
            penetration = cr - face_dists[0][0]
        if penetration <= eps:
            return None
        # convert normal back to world coords
        cos_f = math.cos(square['orientation'])
        sin_f = math.sin(square['orientation'])
        nx = local_nx * cos_f - local_ny * sin_f
        ny = local_nx * sin_f + local_ny * cos_f
        mtv = (nx * penetration, ny * penetration)
        return mtv, (-mtv[0], -mtv[1])

    d = math.sqrt(d2)
    if d >= cr - eps:
        return None
    penetration = cr - d
    # local normal from square point to circle center
    local_nx = diff_x / d
    local_ny = diff_y / d
    # convert to world normal
    cos_f = math.cos(square['orientation'])
    sin_f = math.sin(square['orientation'])
    nx = local_nx * cos_f - local_ny * sin_f
    ny = local_nx * sin_f + local_ny * cos_f
    mtv = (nx * penetration, ny * penetration)
    return mtv, (-mtv[0], -mtv[1])

def project_polygon(axis, corners):
    return [c[0]*axis[0] + c[1]*axis[1] for c in corners]

def get_mtv_square_square(p1, p2, eps=1e-9):
    c1 = get_corners(p1)
    c2 = get_corners(p2)
    axes = [
        (math.cos(p1['orientation']), math.sin(p1['orientation'])),
        (math.cos(p1['orientation'] + math.pi/2), math.sin(p1['orientation'] + math.pi/2)),
        (math.cos(p2['orientation']), math.sin(p2['orientation'])),
        (math.cos(p2['orientation'] + math.pi/2), math.sin(p2['orientation'] + math.pi/2)),
    ]

    min_overlap = float('inf')
    best_axis = None
    best_dir = 1

    for ax in axes:
        proj1 = project_polygon(ax, c1)
        proj2 = project_polygon(ax, c2)
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        # separating axis?
        if max1 < min2 + eps or max2 < min1 + eps:
            return None

        # overlap amount on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = ax
            # determine sign: if center1 proj < center2 proj then p1 is "left" and must go negative along axis
            c1_proj = p1['x'] * ax[0] + p1['y'] * ax[1]
            c2_proj = p2['x'] * ax[0] + p2['y'] * ax[1]
            best_dir = -1 if c1_proj < c2_proj else 1

    # mtv to move p1 by (world axis * min_overlap * best_dir)
    axx, axy = best_axis
    mtv = (axx * (min_overlap + eps) * best_dir, axy * (min_overlap + eps) * best_dir)
    return mtv, (-mtv[0], -mtv[1])

# =============================================================================
# Generalized Triangle Vertex Generation & Geometry Helpers
# =============================================================================

def create_equilateral_triangle_vertices(r):
    """
    Equilateral triangle with circumradius r.
    Vertices placed at angles 90°, 210°, 330°.
    Centroid is at (0, 0).
    """
    base_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
    return [(r * math.cos(a), r * math.sin(a)) for a in base_angles]

def create_right_triangle_vertices(base, height):
    """
    Right-angled triangle with given base and height.
    Uncentered vertices: (0, 0), (base, 0), (0, height).
    Centroid is at (base/3, height/3).
    Centered so average of vertices is (0, 0).
    """
    cx = base / 3.0
    cy = height / 3.0
    return [
        (-cx, -cy),
        (base - cx, -cy),
        (-cx, height - cy)
    ]

def create_isosceles_triangle_vertices(base, height):
    """
    Isosceles triangle with given base and height.
    Uncentered vertices: (-base/2, 0), (base/2, 0), (0, height).
    Centroid is at (0, height/3).
    Centered so average of vertices is (0, 0).
    """
    cy = height / 3.0
    return [
        (-base / 2.0, -cy),
        (base / 2.0, -cy),
        (0.0, height - cy)
    ]

def create_scalene_triangle_vertices(a, b, c):
    """
    Scalene triangle with side lengths a, b, c (where c is base, b is left side).
    Validates triangle inequality: a+b>c, a+c>b, b+c>a.
    Centered so average of vertices is (0, 0).
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(f"Triangle side lengths must be strictly positive: a={a}, b={b}, c={c}")
    if (a + b <= c) or (a + c <= b) or (b + c <= a):
        raise ValueError(
            f"Invalid scalene side lengths ({a}, {b}, {c}) do not satisfy the triangle inequality: "
            f"the sum of any two sides must be strictly greater than the third side."
        )
    
    # Side c lies on x-axis from (0,0) to (c,0)
    # Law of Cosines for angle at vertex 0: a^2 = b^2 + c^2 - 2*b*c*cos(alpha)
    cos_alpha = (b**2 + c**2 - a**2) / (2.0 * b * c)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)
    
    p0 = (0.0, 0.0)
    p1 = (float(c), 0.0)
    p2 = (b * math.cos(alpha), b * math.sin(alpha))
    
    cx = (p0[0] + p1[0] + p2[0]) / 3.0
    cy = (p0[1] + p1[1] + p2[1]) / 3.0
    
    return [
        (p0[0] - cx, p0[1] - cy),
        (p1[0] - cx, p1[1] - cy),
        (p2[0] - cx, p2[1] - cy)
    ]

# =============================================================================
# Generic Polygon & Superellipse Geometry Helpers
# =============================================================================

def get_polygon_bounding_radius(local_vertices):
    """Returns maximum distance from centroid (0,0) to any local vertex."""
    return max(math.hypot(vx, vy) for vx, vy in local_vertices)

# Backward-compatible alias
get_triangle_bounding_radius = get_polygon_bounding_radius

def get_polygon_area(local_vertices):
    """
    Returns exact area of any 2D polygon using the Shoelace formula.
    Works for any number of vertices N >= 3.
    """
    n = len(local_vertices)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, y1 = local_vertices[i]
        x2, y2 = local_vertices[(i + 1) % n]
        area2 += (x1 * y2 - x2 * y1)
    return 0.5 * abs(area2)

# Backward-compatible alias
get_triangle_area = get_polygon_area

def get_polygon_self_inertia_ratio(local_vertices):
    """
    Returns I_self / m (self radius of gyration squared) about centroid for any planar polygon.
    Computed via second area moments: (I_xx + I_yy) / Area.
    Falls back to triangle formula if 3 vertices are passed.
    """
    n = len(local_vertices)
    if n == 3:
        (x0, y0), (x1, y1), (x2, y2) = local_vertices
        s01_sq = (x1 - x0)**2 + (y1 - y0)**2
        s12_sq = (x2 - x1)**2 + (y2 - y1)**2
        s20_sq = (x0 - x2)**2 + (y0 - y2)**2
        return (s01_sq + s12_sq + s20_sq) / 36.0
    
    # Generic polygon area and second moments
    area6 = 0.0
    ixx12 = 0.0
    iyy12 = 0.0
    for i in range(n):
        x1, y1 = local_vertices[i]
        x2, y2 = local_vertices[(i + 1) % n]
        cross = (x1 * y2 - x2 * y1)
        area6 += 3.0 * cross
        ixx12 += cross * (y1 * y1 + y1 * y2 + y2 * y2)
        iyy12 += cross * (x1 * x1 + x1 * x2 + x2 * x2)
    
    if abs(area6) < 1e-12:
        return 0.0
    area = abs(area6) / 6.0
    i_pol = (abs(ixx12) + abs(iyy12)) / 12.0
    return i_pol / area

# Backward-compatible alias
get_triangle_self_inertia_ratio = get_polygon_self_inertia_ratio

def get_polygon_corners(particle):
    """
    Returns the world-space vertices of any convex polygon particle.
    Transforms stored local_vertices by particle['orientation'] and (particle['x'], particle['y']).
    Works for any vertex count.
    """
    cx, cy = particle['x'], particle['y']
    orientation = particle.get('orientation', 0.0)
    
    if 'local_vertices' in particle and particle['local_vertices'] is not None:
        cos_t = math.cos(orientation)
        sin_t = math.sin(orientation)
        corners = []
        for lx, ly in particle['local_vertices']:
            wx = cx + cos_t * lx - sin_t * ly
            wy = cy + sin_t * lx + cos_t * ly
            corners.append((wx, wy))
        return corners
    
    # Fallback for legacy equilateral triangle representation
    r = particle['r']
    base_angles = [math.pi/2, math.pi/2 + 2*math.pi/3, math.pi/2 + 4*math.pi/3]
    return [(cx + r * math.cos(a + orientation), cy + r * math.sin(a + orientation)) for a in base_angles]

# Backward-compatible alias
get_triangle_corners = get_polygon_corners

def create_superellipse_vertices(a, b, n, num_samples=24):
    """
    Generates local vertices approximating a superellipse defined by:
        |x/a|^n + |y/b|^n = 1
    
    Parameters:
    - a: semi-major / semi-minor axis length along x (a > 0)
    - b: semi-major / semi-minor axis length along y (b > 0)
    - n: shape exponent. Must be >= 1.0 to guarantee convexity (convex polygon).
         n = 2: ellipse
         n > 2: squircle / flat-sided rounded box
         1 <= n < 2: rounded diamond / star-like convex shape
    - num_samples: number of polygon vertices sampled around 2*pi (default 24).
         Note: A flatter, higher-exponent superellipse benefits from more sample
         points to represent its flat sides and sharp corners well, while a near-circular
         or low-exponent shape needs fewer sample points.
    
    Returns:
    - List of (x, y) tuples centered on the origin (0, 0).
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Superellipse semi-axes must be strictly positive: a={a}, b={b}")
    if n < 1.0:
        raise ValueError(
            f"Superellipse shape exponent n must be >= 1.0 for convex shapes, but got n={n}. "
            f"Exponents n < 1 produce concave shapes that cannot be handled by convex SAT."
        )
    if num_samples < 4:
        raise ValueError(f"num_samples must be at least 4, got {num_samples}")

    power = 2.0 / n
    vertices = []
    
    for i in range(num_samples):
        theta = 2.0 * math.pi * i / num_samples
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Signed n-th-root adjusted power: sgn(v) * |v|^(2/n)
        sgn_cos = 1.0 if cos_t > 0 else (-1.0 if cos_t < 0 else 0.0)
        sgn_sin = 1.0 if sin_t > 0 else (-1.0 if sin_t < 0 else 0.0)
        
        x = a * sgn_cos * (abs(cos_t) ** power)
        y = b * sgn_sin * (abs(sin_t) ** power)
        vertices.append((x, y))
        
    # Recenter vertices as a safety measure against tiny floating-point asymmetry
    mean_x = sum(v[0] for v in vertices) / float(num_samples)
    mean_y = sum(v[1] for v in vertices) / float(num_samples)
    centered_vertices = [(v[0] - mean_x, v[1] - mean_y) for v in vertices]
    
    return centered_vertices

def get_mtv_polygon_polygon(p1, p2, eps=1e-9):
    """
    Generic Separating Axis Theorem (SAT) collision test for any two convex polygons.
    Dynamically loops over however many vertices each shape has (corners list).
    Returns (mtv_for_p1, mtv_for_p2) or None if not colliding.
    """
    c1 = get_polygon_corners(p1)
    c2 = get_polygon_corners(p2)

    def edge_normals(corners):
        normals = []
        n = len(corners)
        for i in range(n):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % n]
            edge_x, edge_y = x2 - x1, y2 - y1
            length = math.hypot(edge_x, edge_y)
            if length < eps:
                continue
            # perpendicular, normalized
            normals.append((-edge_y / length, edge_x / length))
        return normals

    axes = edge_normals(c1) + edge_normals(c2)

    min_overlap = float('inf')
    best_axis = None

    for ax in axes:
        proj1 = project_polygon(ax, c1)
        proj2 = project_polygon(ax, c2)
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return None  # separating axis found -> no collision

        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = ax

    axx, axy = best_axis
    # sign correction: push p1 away from p2 along best_axis
    c1_proj = p1['x'] * axx + p1['y'] * axy
    c2_proj = p2['x'] * axx + p2['y'] * axy
    direction = -1 if c1_proj < c2_proj else 1

    mtv = (axx * (min_overlap + eps) * direction, axy * (min_overlap + eps) * direction)
    return mtv, (-mtv[0], -mtv[1])

def get_mtv_polygon_circle(polygon, circle, eps=1e-9):
    """
    Separating Axis Theorem (SAT) collision test between a convex polygon and a circle.
    Candidate axes:
      1. Polygon edge normals
      2. Axis from polygon center to circle center
      3. Axes from polygon vertices to circle center (for corner contacts)
    Returns (mtv_for_polygon, mtv_for_circle) or None if not colliding.
    The MTV is oriented to push the polygon away from the circle.
    """
    corners = get_polygon_corners(polygon)
    if not corners:
        return None

    cr = circle['r']
    cx_poly, cy_poly = polygon['x'], polygon['y']
    cx_circ, cy_circ = circle['x'], circle['y']

    # 1. Broad-phase Bounding Circle check
    dx = cx_circ - cx_poly
    dy = cy_circ - cy_poly
    dist_sq = dx*dx + dy*dy
    r_bound_poly = polygon.get('r_bound', polygon.get('r', 10.0))
    safe_dist = r_bound_poly + cr + eps
    if dist_sq > safe_dist**2:
        return None

    # 2. Build candidate axes
    axes = []
    n = len(corners)

    # Edge normals
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        edge_x, edge_y = x2 - x1, y2 - y1
        length = math.hypot(edge_x, edge_y)
        if length > eps:
            axes.append((-edge_y / length, edge_x / length))

    # Center-to-center axis
    center_dist = math.hypot(dx, dy)
    if center_dist > eps:
        axes.append((dx / center_dist, dy / center_dist))
    else:
        axes.append((1.0, 0.0))

    # Vertex-to-circle-center axes
    for vx, vy in corners:
        v_dx = cx_circ - vx
        v_dy = cy_circ - vy
        v_len = math.hypot(v_dx, v_dy)
        if v_len > eps:
            axes.append((v_dx / v_len, v_dy / v_len))

    # 3. Test for separating axes
    min_overlap = float('inf')
    best_axis = None

    for ax in axes:
        # Project polygon corners
        proj_poly = [c[0] * ax[0] + c[1] * ax[1] for c in corners]
        min_poly, max_poly = min(proj_poly), max(proj_poly)

        # Project circle
        proj_circ = cx_circ * ax[0] + cy_circ * ax[1]
        min_circ = proj_circ - cr
        max_circ = proj_circ + cr

        if max_poly < min_circ + eps or max_circ < min_poly + eps:
            return None  # Separating axis found -> no collision

        overlap = min(max_poly, max_circ) - max(min_poly, min_circ)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = ax

    # 4. Sign correction: push polygon away from circle
    axx, axy = best_axis
    c_poly_proj = cx_poly * axx + cy_poly * axy
    c_circ_proj = cx_circ * axx + cy_circ * axy
    direction = -1 if c_poly_proj < c_circ_proj else 1

    mtv = (axx * (min_overlap + eps) * direction, axy * (min_overlap + eps) * direction)
    return mtv, (-mtv[0], -mtv[1])

# Aliases
get_mtv_triangle_triangle = get_mtv_polygon_polygon
get_mtv_triangle_circle = get_mtv_polygon_circle
get_mtv_circle_triangle = lambda c, t, eps=1e-9: (
    (lambda res: (res[1], res[0]) if res else None)(get_mtv_triangle_circle(t, c, eps))
)

# ---------- shape-aware cheap reject ----------
def quick_center_aabb_reject(p1, p2, margin=0.0):
    dx = abs(p2['x'] - p1['x'])
    dy = abs(p2['y'] - p1['y'])

    # circle-circle
    if p1['shape'] == 'circle' and p2['shape'] == 'circle':
        max_extent = (p1['r'] + p2['r']) + margin
        return dx > max_extent or dy > max_extent

    # square-square
    if p1['shape'] == 'square' and p2['shape'] == 'square':
        max_extent = (abs(p1['r']) + abs(p2['r'])) * math.sqrt(2) + margin
        return dx > max_extent or dy > max_extent

    # circle-square either order
    if p1['shape'] == 'circle' and p2['shape'] == 'square':
        circle_r = p1['r']; square_half = p2['r']
    elif p1['shape'] == 'square' and p2['shape'] == 'circle':
        circle_r = p2['r']; square_half = p1['r']
    else:
        max_extent = (abs(p1['r']) + abs(p2['r'])) * math.sqrt(2) + margin
        return dx > max_extent or dy > max_extent

    max_extent = circle_r + square_half * math.sqrt(2) + margin
    return dx > max_extent or dy > max_extent

# ---------- robust iterative resolve ----------
def resolveCollisions(particles, max_iters=12, tol=1e-6):
    """
    Iteratively separate all overlapping pairs using per-pair MTVs.
    Returns the list of particles (positions modified in-place).
    - max_iters: maximum relaxation iterations (12 is usually enough).
    - tol: small epsilon for comparisons.
    """
    n = len(particles)
    if n < 2:
        return []

    for it in range(max_iters):
        any_collision = False
        # accumulate translations to apply this iteration
        #translations = [(0.0, 0.0) for _ in range(n)]
        #counts = [0 for _ in range(n)]  # how many MTV contributions a particle gets (to average)

        for i in range(n):
            p1 = particles[i]
            for j in range(i+1, n):
                p2 = particles[j]

                # quick reject
                if quick_center_aabb_reject(p1, p2):
                    continue

                # get MTV for this pair (returns pair of mtvs corresponding to p1 and p2)
                mtv_pair = None
                if p1['shape'] == 'circle' and p2['shape'] == 'circle':
                    mtv_pair = get_mtv_circle_circle(p1, p2, eps=tol)
                elif p1['shape'] == 'square' and p2['shape'] == 'square':
                    mtv_pair = get_mtv_square_square(p1, p2, eps=tol)
                elif p1['shape'] == 'circle' and p2['shape'] == 'square':
                    mtv_pair = get_mtv_circle_square(p1, p2, eps=tol)
                elif p1['shape'] == 'square' and p2['shape'] == 'circle':
                    # get mtv with circle first; reorder results to match (p1,p2)
                    res = get_mtv_circle_square(p2, p1, eps=tol)
                    if res is not None:
                        # res is (mtv_circle, mtv_square) where circle was p2
                        mtv_pair = (res[1], res[0])  # convert to (mtv_for_p1, mtv_for_p2)
                else:
                    continue

                if mtv_pair is None:
                    continue

                any_collision = True
                mtv1, mtv2 = mtv_pair

                factor1 = p2['r']**2/(p1['r']**2 + p2['r']**2) 
                factor2 = p1['r']**2/(p1['r']**2 + p2['r']**2)
                p1['x'] += factor1 * mtv1[0]; p1['y'] += factor1 * mtv1[1]
                p2['x'] += factor2 * mtv2[0]; p2['y'] += factor2 * mtv2[1]


        if not any_collision:
            # all clear
            break

        # apply averaged translations, but apply only a fraction (0.5) to avoid oscillation in extreme cases
        # for k in range(n):
        #     if counts[k] > 0:
        #         avg_tx = translations[k][0] / counts[k]
        #         avg_ty = translations[k][1] / counts[k]
        #         particles[k]['x'] += 0.5 * avg_tx
        #         particles[k]['y'] += 0.5 * avg_ty

    # after iterations, return list of particles that were involved in at least one collision
    impacted = []
    for i in range(n):
        for j in range(i+1, n):
            # final check: if still overlapping, include them
            if particles[i]['shape'] == 'circle' and particles[j]['shape'] == 'circle':
                d2 = (particles[i]['x'] - particles[j]['x'])**2 + (particles[i]['y'] - particles[j]['y'])**2
                if d2 < (particles[i]['r'] + particles[j]['r'])**2 - tol:
                    impacted.append(particles[i]); impacted.append(particles[j])
            elif particles[i]['shape'] == 'square' and particles[j]['shape'] == 'square':
                if get_mtv_square_square(particles[i], particles[j], eps=tol) is not None:
                    impacted.append(particles[i]); impacted.append(particles[j])
            else:
                # circle-square cases
                a, b = (particles[i], particles[j]) if particles[i]['shape']=='circle' else (particles[j], particles[i])
                if get_mtv_circle_square(a, b, eps=tol) is not None:
                    impacted.append(particles[i]); impacted.append(particles[j])
    # dedupe impacted
    unique_impacted = []
    seen = set()
    for p in impacted:
        sid = id(p)
        if sid not in seen:
            seen.add(sid)
            unique_impacted.append(p)
    return unique_impacted
