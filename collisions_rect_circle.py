"""
collisions_rect_circle.py - Collision detection for rectangles and circles
Supports: rectangle-rectangle, circle-circle, and rectangle-circle collisions
"""
import math

def dist(p1, p2):
    """Calculate Euclidean distance between two particles."""
    return math.hypot(p2['x'] - p1['x'], p2['y'] - p1['y'])

def get_corners(p):
    """Calculates the 4 corners of a rotated rectangle."""
    if p['shape'] != 'rectangle':
        return []
        
    w, h = p['w'], p['h']
    
    c = math.cos(p.get('orientation', 0.0))
    s = math.sin(p.get('orientation', 0.0))
    
    dx1, dy1 = w/2 * c, w/2 * s
    dx2, dy2 = -h/2 * s, h/2 * c
    
    return [
        (p['x'] + dx1 + dx2, p['y'] + dy1 + dy2), # Top Right
        (p['x'] - dx1 + dx2, p['y'] - dy1 + dy2), # Top Left
        (p['x'] - dx1 - dx2, p['y'] - dy1 - dy2), # Bottom Left
        (p['x'] + dx1 - dx2, p['y'] + dy1 - dy2)  # Bottom Right
    ]

def quick_center_aabb_reject(p1, p2, margin=0.0):
    """Fast circular reject using squared distances to avoid math.sqrt."""
    dist_sq = (p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2
    safe_dist = p1['bounding_r'] + p2['bounding_r'] + margin
    return dist_sq > safe_dist**2

def project_polygon(axis, corners):
    """Projects polygon corners onto an axis for SAT."""
    min_p = max_p = corners[0][0]*axis[0] + corners[0][1]*axis[1]
    for c in corners[1:]:
        p = c[0]*axis[0] + c[1]*axis[1]
        if p < min_p: min_p = p
        if p > max_p: max_p = p
    return min_p, max_p

def get_mtv_square_square(p1, p2, eps=1e-9):
    """
    Highly optimized SAT for rotated rectangles.
    Checks only the 4 unique normal axes and uses angles directly.
    """
    # Only for rectangle-rectangle collisions
    if p1['shape'] != 'rectangle' or p2['shape'] != 'rectangle':
        return None
        
    # 1. Broad-phase Bounding Circle check FIRST
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    dist_sq = dx*dx + dy*dy
    safe_dist = p1['bounding_r'] + p2['bounding_r'] + eps
    
    if dist_sq > safe_dist**2:
        return None # Too far apart, no collision
        
    # 2. Get corners
    corners1 = get_corners(p1)
    corners2 = get_corners(p2)
    
    # 3. Get the 4 unique orthogonal axes (2 from p1, 2 from p2)
    o1 = p1.get('orientation', 0.0)
    o2 = p2.get('orientation', 0.0)
    
    c1, s1 = math.cos(o1), math.sin(o1)
    c2, s2 = math.cos(o2), math.sin(o2)
    
    axes = [
        (c1, s1),   # Local X axis of p1
        (-s1, c1),  # Local Y axis of p1
        (c2, s2),   # Local X axis of p2
        (-s2, c2)   # Local Y axis of p2
    ]
    
    overlap = float('inf')
    smallest_axis = None
    
    # 4. Project corners onto the 4 axes
    for ax, ay in axes:
        # Inline projection for p1
        min1 = max1 = corners1[0][0]*ax + corners1[0][1]*ay
        for i in range(1, 4):
            proj = corners1[i][0]*ax + corners1[i][1]*ay
            if proj < min1: min1 = proj
            elif proj > max1: max1 = proj
            
        # Inline projection for p2
        min2 = max2 = corners2[0][0]*ax + corners2[0][1]*ay
        for i in range(1, 4):
            proj = corners2[i][0]*ax + corners2[i][1]*ay
            if proj < min2: min2 = proj
            elif proj > max2: max2 = proj
            
        # 5. Check for a separating gap
        if max1 < min2 or max2 < min1:
            return None # A separating axis was found!
            
        # 6. Track the smallest overlap
        o = min(max1, max2) - max(min1, min2)
        if o < overlap:
            overlap = o
            smallest_axis = (ax, ay)
            
    # 7. Ensure the MTV pushes p1 AWAY from p2
    if (dx * smallest_axis[0] + dy * smallest_axis[1]) < 0:
        smallest_axis = (-smallest_axis[0], -smallest_axis[1])
        
    mtv_x = smallest_axis[0] * overlap
    mtv_y = smallest_axis[1] * overlap
    
    return (mtv_x, mtv_y), (-mtv_x, -mtv_y)

def get_mtv_circle_circle(c1, c2, eps=1e-9):
    """
    Check collision between two circles and return MTV.
    Returns: ((mtv_x, mtv_y), (-mtv_x, -mtv_y)) or None
    """
    if c1['shape'] != 'circle' or c2['shape'] != 'circle':
        return None
        
    dx = c1['x'] - c2['x']
    dy = c1['y'] - c2['y']
    dist_sq = dx*dx + dy*dy
    r_sum = c1['radius'] + c2['radius'] + eps
    
    if dist_sq >= r_sum**2:
        return None  # No collision
        
    dist = math.sqrt(dist_sq)
    if dist == 0:
        # Circles are exactly on top of each other - push in arbitrary direction
        dist = 0.001
        dx, dy = 1.0, 0.0
    else:
        dx, dy = dx/dist, dy/dist
        
    overlap = r_sum - dist
    mtv_x = dx * overlap
    mtv_y = dy * overlap
    
    return (mtv_x, mtv_y), (-mtv_x, -mtv_y)

def get_mtv_rectangle_circle(rect, circle, eps=1e-9):
    """
    Check collision between a rectangle and a circle using SAT.
    Returns: ((mtv_x, mtv_y), (-mtv_x, -mtv_y)) or None
    The MTV is oriented to push the rectangle away from the circle.
    """
    if rect['shape'] != 'rectangle' or circle['shape'] != 'circle':
        return None
        
    # Broad phase check
    dx = rect['x'] - circle['x']
    dy = rect['y'] - circle['y']
    dist_sq = dx*dx + dy*dy
    safe_dist = rect['bounding_r'] + circle['radius'] + eps
    
    if dist_sq > safe_dist**2:
        return None
        
    # Get rectangle corners
    corners = get_corners(rect)
    
    # Test axes: rectangle's two edge normals + axis from rect center to circle center
    o = rect.get('orientation', 0.0)
    c, s = math.cos(o), math.sin(o)
    
    # Rectangle's axes
    axes = [
        (c, s),      # Local X axis
        (-s, c),     # Local Y axis
    ]
    
    # Add axis from rect center to circle center (if not at same point)
    center_dist = math.sqrt(dx*dx + dy*dy)
    if center_dist > 1e-9:
        axes.append((dx/center_dist, dy/center_dist))
    else:
        axes.append((1.0, 0.0))
        
    overlap = float('inf')
    smallest_axis = None
    
    for ax, ay in axes:
        # Project rectangle corners
        min_rect = max_rect = corners[0][0]*ax + corners[0][1]*ay
        for i in range(1, 4):
            proj = corners[i][0]*ax + corners[i][1]*ay
            if proj < min_rect: min_rect = proj
            elif proj > max_rect: max_rect = proj
            
        # Project circle (center ± radius along axis)
        circle_center_proj = circle['x']*ax + circle['y']*ay
        min_circle = circle_center_proj - circle['radius']
        max_circle = circle_center_proj + circle['radius']
        
        # Check for separation
        if max_rect < min_circle or max_circle < min_rect:
            return None
            
        # Track smallest overlap
        o = min(max_rect, max_circle) - max(min_rect, min_circle)
        if o < overlap:
            overlap = o
            smallest_axis = (ax, ay)
            
    # Ensure MTV pushes rectangle away from circle
    if (dx * smallest_axis[0] + dy * smallest_axis[1]) < 0:
        smallest_axis = (-smallest_axis[0], -smallest_axis[1])
        
    mtv_x = smallest_axis[0] * overlap
    mtv_y = smallest_axis[1] * overlap
    
    return (mtv_x, mtv_y), (-mtv_x, -mtv_y)

def get_mtv_any_any(p1, p2, eps=1e-9):
    """
    Universal collision detection that routes to the appropriate method.
    Handles: rectangle-rectangle, circle-circle, rectangle-circle.
    """
    if p1['shape'] == 'rectangle' and p2['shape'] == 'rectangle':
        return get_mtv_square_square(p1, p2, eps)
    elif p1['shape'] == 'circle' and p2['shape'] == 'circle':
        return get_mtv_circle_circle(p1, p2, eps)
    elif p1['shape'] == 'rectangle' and p2['shape'] == 'circle':
        return get_mtv_rectangle_circle(p1, p2, eps)
    elif p1['shape'] == 'circle' and p2['shape'] == 'rectangle':
        # Flip the order and negate the result
        result = get_mtv_rectangle_circle(p2, p1, eps)
        if result is None:
            return None
        mtv1, mtv2 = result
        return (-mtv1[0], -mtv1[1]), (-mtv2[0], -mtv2[1])
    else:
        return None
