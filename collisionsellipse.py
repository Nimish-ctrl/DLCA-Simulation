import math
import random

def dist(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def generate_ellipse_circles(x, y, a, b, angle):
    """
    Places circles along the major axis of the ellipse at relative positions `positions`.
    Each circle radius is the local minor-axis half-width at that major-axis offset.
    """
    circles = []
    # same sample positions you used
    positions = [-0.65, -0.325, 0, 0.325, 0.65]
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)

    for t in positions:
        local_x = t * a               # coordinate along major axis (unrotated)
        # rotate the major-axis offset to world coords
        rot_x = local_x * cos_t
        rot_y = local_x * sin_t
        pos_x = x + rot_x
        pos_y = y + rot_y
        # radius is the local minor-axis cross-section: b * sqrt(1 - (local_x/a)^2)
        frac = t  # t = local_x / a already
        radius = b * math.sqrt(max(0.0, 1.0 - frac * frac))
        # avoid zero-radius circles (very small), clamp to a tiny positive value
        radius = max(radius, 1e-6)
        circles.append(((pos_x, pos_y), radius))

    return circles


def ellipseCollisions(e1, e2, max_iter_per_pair=5):
    """
    Check and resolve collisions between the circles approximating two ellipses.
    - Modifies e1['x'], e1['y'], e2['x'], e2['y'] in-place.
    - Recomputes e1['circles'] and e2['circles'] after changes.
    - Returns True if any collision was detected and (attempted) resolved.
    """
    # ensure circles lists exist (generate if missing)
    c1 = e1.get('circles') or generate_ellipse_circles(e1['x'], e1['y'],
                                                     e1['major_radius'],
                                                     e1['minor_radius'],
                                                     e1['angle'])
    c2 = e2.get('circles') or generate_ellipse_circles(e2['x'], e2['y'],
                                                     e2['major_radius'],
                                                     e2['minor_radius'],
                                                     e2['angle'])

    collided_any = False

    # loop through ALL pairs and attempt to resolve; do a few iterations per pair
    for idx1, (p1, r1) in enumerate(c1):
        for idx2, (p2, r2) in enumerate(c2):
            # compute squared distance
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist_sq = dx * dx + dy * dy
            min_dist = r1 + r2

            # if overlapping, try to resolve iteratively
            iter_count = 0
            while dist_sq < (min_dist * min_dist) and iter_count < max_iter_per_pair:
                iter_count += 1
                collided_any = True

                # compute current distance (avoid sqrt(0) dividing by zero)
                if dist_sq > 1e-12:
                    dist_val = math.sqrt(dist_sq)
                    nx, ny = dx / dist_val, dy / dist_val
                else:
                    # identical centers: pick a random small unit vector
                    theta = random.uniform(0, 2 * math.pi)
                    nx, ny = math.cos(theta), math.sin(theta)
                    dist_val = 1e-6  # tiny

                # overlap amount
                overlap = min_dist - dist_val
                if overlap <= 0:
                    break

                # base shift distributes overlap equally
                base_shift = overlap * 0.5

                # cap: prevent huge teleportation, but allow enough to resolve
                # choose cap as a fraction of min_dist (safe and adaptive)
                cap = max(0.1, 0.4 * min_dist)  # at least 0.1 px or 40% of min_dist
                shift = min(base_shift, cap)

                # apply shifts: move e1 in +n, e2 in -n (you can reverse sign depending on your coordinate system)
                e1['x'] += nx * shift
                e1['y'] += ny * shift
                e2['x'] -= nx * shift
                e2['y'] -= ny * shift

                # regenerate circle lists because centers changed
                c1 = generate_ellipse_circles(e1['x'], e1['y'],
                                             e1['major_radius'], e1['minor_radius'], e1['angle'])
                c2 = generate_ellipse_circles(e2['x'], e2['y'],
                                             e2['major_radius'], e2['minor_radius'], e2['angle'])

                # update current pair's p1/p2/r1/r2 because circles moved
                p1, r1 = c1[idx1]
                p2, r2 = c2[idx2]

                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dist_sq = dx * dx + dy * dy
                min_dist = r1 + r2

            # continue to next pair (we do NOT return immediately; resolve all pairs)
    # store updated circles back into the objects (so caller can use them)
    e1['circles'] = generate_ellipse_circles(e1['x'], e1['y'],
                                             e1['major_radius'], e1['minor_radius'], e1['angle'])
    e2['circles'] = generate_ellipse_circles(e2['x'], e2['y'],
                                             e2['major_radius'], e2['minor_radius'], e2['angle'])

    return collided_any


def resolveCollisions(particles):
    """
    Broad-phase + narrow-phase resolution.
    Returns True if any collision was detected/resolved.
    """
    detected = False
    n = len(particles)
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = particles[i], particles[j]
            # safer broadphase: use max radius as approximate bounding radius
            bound_x = abs(p1['x'] - p2['x'])
            bound_y = abs(p1['y'] - p2['y'])
            approx_bound = (max(p1['major_radius'], p1['minor_radius']) +
                            max(p2['major_radius'], p2['minor_radius']))
            if bound_x <= approx_bound and bound_y <= approx_bound:
                if ellipseCollisions(p1, p2):
                    detected = True
    return detected
