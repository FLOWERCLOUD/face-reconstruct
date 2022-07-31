import numpy as np


def convex_hull_2(pts):
    ''' find convex hull given a set of 2d points
    
    Parameters
    ---------------
    pts
        nx2 2d points
    
    Return
    --------------
    idxpts
        indices of points in pts, ordered such that it forms a polygon.
        The indices has end point duplication.
    '''
    import scipy.spatial as spatial

    hull = spatial.ConvexHull(pts)
    idxpts = hull.vertices
    res = np.append(idxpts, idxpts[0])
    return res


def rotation_matrix_by_X(angle_rad):
    ''' rotation matrix that rotates an object around X by angle_rad
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian
    '''
    x = angle_rad
    rs = [1, 0, 0,
          0, np.cos(x), -np.sin(x),
          0, np.sin(x), np.cos(x)]
    rotmat = np.reshape(rs, (3, 3)).T
    return rotmat


def rotation_matrix_by_Y(angle_rad):
    ''' rotation matrix that rotates an object around Y by angle_rad
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a tensor
    '''
    x = angle_rad

    rs = [np.cos(x), 0, np.sin(x),
          0, 1, 0,
          -np.sin(x), 0, np.cos(x)]
    rotmat = np.reshape(rs, (3, 3)).T
    return rotmat


def rotation_matrix_by_Z(angle_rad):
    ''' rotation matrix that rotates an object around Z by angle_rad.
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    angle_rad
        the rotation angle in radian, must be a tensor
    '''
    x = angle_rad
    rs = [np.cos(x), -np.sin(x), 0,
          np.sin(x), np.cos(x), 0,
          0, 0, 1]
    rotmat = np.reshape(rs, (3, 3)).T
    return rotmat


def rotation_matrix_by_xyz(dx, dy, dz):
    ''' create rotation matrix from rotating around x by dx, y by dy and z by z
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    dx,dy,dz
        angles, represented in radian
        
    Return
    --------------
    R
        rotation matrix where point.dot(R) gives a rotated point
    '''
    rx = rotation_matrix_by_X(dx)
    ry = rotation_matrix_by_Y(dy)
    rz = rotation_matrix_by_Z(dz)
    rotmat = rx.dot(ry).dot(rz)
    return rotmat


def rotation_matrix_to_xyz(rotmat):
    ''' convert rotation matrix to rotations around x,y,z axis.
    The converted matrices and the original matrix has the following
    relation: Rx.dot(Ry).dot(Rz) == rotmat
    
    Parameters
    ------------------
    rotmat
        3x3 rotation matrix
        
    Return
    -----------------
    xyz
        1x3 vector of [x,y,z] rotations
    '''
    cos_y_cos_z = rotmat[0, 0]
    cos_y_sin_z = rotmat[0, 1]
    cos_y_sin_x = rotmat[1, 2]
    cos_y_cos_x = rotmat[2, 2]
    sin_y = -rotmat[0, 2]
    y = np.arcsin(sin_y)
    cos_y = np.cos(y)

    if np.abs(cos_y) < 1e-6:
        # assuming y to be very close to pi/2 or -pi/2
        y = np.sign(y) * np.pi / 2
        sin_x_minus_z = rotmat[1, 0]
        cos_x_minus_z = rotmat[1, 1]
        x_minus_z = np.arctan2(sin_x_minus_z, cos_x_minus_z)
        z = 0
        x = x_minus_z
    else:
        sin_z = cos_y_sin_z / cos_y
        cos_z = cos_y_cos_z / cos_y
        z = np.arctan2(sin_z, cos_z)

        sin_x = cos_y_sin_x / cos_y
        cos_x = cos_y_cos_x / cos_y
        x = np.arctan2(sin_x, cos_x)
    return np.array([x, y, z])


def rotation_matrix_by_xyz_vec(xyz):
    ''' create rotation matrix from rotating around x by dx, y by dy and z by z
    The returned rotation matrix is to be multiplied to the right of a point.
    
    Parameters
    ------------------
    xyz
        angles [dx,dy,dz], represented in radian
        
    Return
    --------------
    R
        rotation matrix where point.dot(R) gives a rotated point
    '''
    rx = rotation_matrix_by_X(xyz[0])
    ry = rotation_matrix_by_Y(xyz[1])
    rz = rotation_matrix_by_Z(xyz[2])
    rotmat = rx.dot(ry).dot(rz)
    return rotmat


def decompose_transmat(transmat):
    ''' decompose 4x4 transformation matrix into rotation, scale and translation
    
    Return
    -------------------
    rotmat
        3x3 rotation matrix
    scale
        1x3 scale vector
    translation
        1x3 translation vector
    '''
    A = transmat[:-1, :-1]
    s = np.linalg.norm(A, axis=1)
    rotmat = A / s.reshape((-1, 1))
    translate = transmat[-1, :-1]
    return (rotmat, s, translate)


def laplacian_from_mesh(faces, num_vert=None, normalize=False):
    ''' create laplacian matrix from faces of a mesh
    
    parameter
    ----------------
    faces
        the face array of the mesh, NxK matrix with N faces
    num_vert
        number of vertices in the mesh, if None, it will be found as faces.max()+1
    normalize
        shall we normalize each row of the laplacian matrix so that the positive numbers sumed to 1?
        
    return
    ------------------
    lpmat
        the laplacian matrix
    '''
    if num_vert is None:
        num_vert = faces.max() + 1
    adjmat = adjmat_from_mesh(num_vert, faces)
    lpmat = laplacian_from_adjmat(adjmat, normalize=normalize)
    return lpmat


def laplacian_from_adjmat(adjmat, normalize=False):
    ''' create laplacian matrix from adjmat
    
    Parameters
    --------------------
    adjmat
        adjacency matrix
    normalize
        should we normalize each row of the laplacian matrix so that the positive number of each row sums to 1?
        
    Return
    -------------------
    lpmat
        laplacian matrix
    '''
    import scipy.sparse as sparse
    adjmat = adjmat.astype('float')
    dgs = adjmat.sum(axis=1)
    dgs = np.array(dgs)
    if sparse.issparse(adjmat):
        if normalize:
            inv_dgs = sparse.csr_matrix(1 / dgs)
            L = adjmat.multiply(inv_dgs) - sparse.eye(adjmat.shape[0])
        else:
            L = adjmat - sparse.diags(dgs.flatten(), shape=adjmat.shape)
    else:
        if normalize:
            L = adjmat / dgs - np.eye(len(adjmat))
        else:
            L = adjmat - np.diag(dgs.flatten())
    return L


def adjmat_from_mesh(n, faces):
    ''' create sparse adjacency matrix from a mesh
    
    n = number of vertices
    
    faces = face array
    
    return adjmat, where adjmat[i,j]=True iff ith and jth vertex are connected
    '''
    import scipy.sparse as sparse
    import itertools
    adjmat = sparse.lil_matrix((n, n), dtype=bool)
    for i, j in itertools.combinations(np.arange(faces.shape[-1]), 2):
        u = faces[:, i]
        v = faces[:, j]
        adjmat[u, v] = True
        adjmat[v, u] = True
    return adjmat


def submesh_by_vertex(idxvts, faces):
    ''' extract a submesh consisting of a set of chosen vertices.
    
    Parameters
    ------------------
    idxvts
        indices of vertices in the submesh, indices must be unique
    faces
        faces of the mesh
        
    Return
    -------------------
    subfaces
        a nx3 face array defining the faces of the new mesh.
        The vertex subface[i,j]==k refers to the vertex idxvts[k]
    '''

    # check uniqueness
    x = np.unique(idxvts)
    assert len(x) == len(idxvts), 'idxvts is not unique'

    idx_old2new = np.zeros(faces.max() + 1, dtype=int)
    idx_old2new[idxvts] = np.arange(len(idxvts))

    # select faces that are composed of vertices in idxvts
    f = faces.flatten()
    maskf_in_idxvts = np.in1d(f, idxvts).reshape(faces.shape)
    mask = np.all(maskf_in_idxvts, axis=1)
    subf = faces[mask]

    subfaces = idx_old2new[subf]
    return subfaces


def submesh_by_faces(faces, idxface):
    ''' extract a submesh consisting a set of selected faces
    
    Parameters
    ----------------------
    faces
        faces of the original model
    idxface
        indices of selected faces
        
    Return
    -------------------------
    idxvts
        indices of vertices in the new mesh
    newfaces
        submesh face array using vertices[idxvts] as vertices
    '''
    # select vertices
    subface = faces[idxface]
    idxvts = np.unique(subface.flatten())

    # re-index the vertices
    idx_old2new = np.zeros(subface.max() + 1, dtype=int)
    idx_old2new[idxvts] = np.arange(len(idxvts))
    f_out = idx_old2new[subface]
    return idxvts, f_out


def find_mesh_boundary_old(faces):
    ''' find mesh boundary given a set of faces
    
    Return
    ------------------------------
    list of paths
        a list of vertex indices, each denoting a path
    '''
    import trimesh
    import scipy.spatial as spatial
    import networkx as nx
    import shortfunc as sf

    # find boundary vertices
    n_vertex = np.max(faces) + 1
    vts = np.random.rand(n_vertex, 3)
    meshobj = trimesh.Trimesh(vts, faces, process=False)
    outline = meshobj.outline()
    boundary_vts = outline.vertices

    # find vertex indices
    kd = spatial.cKDTree(vts)
    _, idx = kd.query(boundary_vts)

    # find all paths
    pathlist = []
    for sg in nx.connected_component_subgraphs(outline.vertex_graph):
        links = list(nx.eulerian_circuit(sg))
        p = np.array([x[0] for x in links])
        pathlist.append(idx[p])

    return pathlist


def find_mesh_boundary_edges(faces):
    ''' find mesh boundary edges given a set of faces
    
    Return
    ------------------------------
    edges
        nx2, each row is an edge belong to the mesh boundary
    '''
    import shortfunc as sf

    # find boundary edges
    face_sort = np.sort(faces, axis=1)
    edges = np.row_stack((face_sort[:, [0, 1]], face_sort[:, [0, 2]], face_sort[:, [1, 2]]))
    edges, _, _, counts = sf.unique_rows(edges)
    edges = edges[counts == 1]

    return edges


def find_longest_mesh_boundary(vertices, faces, close_path=False):
    ''' find the longest mesh boundary given the vertex and face array
    
    Parameters
    -----------------
    vertices
        vertices of the mesh
    faces
        face array of the mesh
    close_path
        should the returned path be closed?
        A closed path have v[0]==v[-1]
    
    return
    ----------------------
    idxv_path
        indices of the vertices on the longest mesh boundary.
    '''
    plist = find_mesh_boundary(faces)
    plen = np.zeros(len(plist))
    for i in range(len(plist)):
        idxv = plist[i]
        idxv = np.append(idxv, idxv[0])
        v0 = vertices[idxv[:-1]]
        v1 = vertices[idxv[1:]]
        lens = np.linalg.norm(v0 - v1, axis=1)
        plen[i] = lens.sum()
    idxmax = np.argmax(plen)
    return plist[idxmax]


def find_mesh_boundary(faces):
    ''' find mesh boundary given a set of faces
    
    Return
    ------------------------------
    list of paths
        a list of vertex indices, each denoting a path.
        Note that path[i][0] != path[i][-1], 
        but the path is implicitly closed
    '''
    import trimesh
    import scipy.spatial as spatial
    import networkx as nx
    from . import shortfunc as sf

    # find boundary edges
    face_sort = np.sort(faces, axis=1)
    edges = np.row_stack((face_sort[:, [0, 1]], face_sort[:, [0, 2]], face_sort[:, [1, 2]]))
    edges, _, _, counts = sf.unique_rows(edges)
    edges = edges[counts == 1]

    # create border graph
    g = nx.Graph()
    g.add_edges_from(edges)

    # find all paths
    pathlist = []
    for sg in nx.connected_component_subgraphs(g):
        if nx.is_eulerian(sg):
            links = list(nx.eulerian_circuit(sg))
        else:
            esg = sg.to_directed()
            links = list(nx.eulerian_circuit(esg))
        idxv_in_path = [x[0] for x in links]
        if idxv_in_path[0] == idxv_in_path[-1]:
            idxv_in_path = idxv_in_path[:-1]
        pathlist.append(np.array(idxv_in_path))
    return pathlist

    ## find boundary vertices
    # n_vertex = np.max(faces)+1
    # vts = np.random.rand(n_vertex, 3)
    # meshobj = trimesh.Trimesh(vts, faces, process=False)
    # outline = meshobj.outline()
    # boundary_vts = outline.vertices

    ## find vertex indices
    # kd = spatial.cKDTree(vts)
    # _, idx = kd.query(boundary_vts)

    ## find all paths
    # pathlist = []
    # for sg in nx.connected_component_subgraphs(outline.vertex_graph):
    # links = list(nx.eulerian_circuit(sg))
    # p = np.array([x[0] for x in links])
    # pathlist.append(idx[p])

    # return pathlist


def find_first_commom_v(v_list, find_list, v_befond_index):
    commom_v = list()
    v_befond = v_list[v_befond_index]
    for i in find_list:
        diff_v = v_list[i] - v_befond
        dis = np.sqrt(diff_v[0] ** 2 + diff_v[1] ** 2 + diff_v[2] ** 2)
        if dis < 0.001:
            commom_v.append(i)

    return commom_v


def mesh_distance_transform_v2(faces, idxsites,
                               vertices=None, distance_metric='hop',
                               fillval=np.inf):
    ''' for each vertex in a mesh, find the shortest distance to one of the sites

    Parameters
    -------------------------
    faces
        mesh faces
    idxsites
        indices of vertices to which distances are computed
    vertices
        vertices of the mesh. When distance_metric is 'hop', vertex location is not needed
    distance_metric
        'l2' or 'hop'. 'l2' is the l2 norm, 'hop' is the number of hops from any site to the target vertex
    fillval
        the distance is set to this value if the vertex is not reachable from any site

    Returns
    ----------------------
    distance_array
        an array of the length max(faces)+1, which stores the distance of each vertex.
        Nan if the distance is meaningless (e.g., a vertex not contained in any face).
    '''
    idxsites = np.array(idxsites)
    import networkx as nx
    g = nx.Graph()
    nv = faces.max() + 1
    g.add_nodes_from(np.arange(nv))
    e1, e2, e3 = faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]
    edges = np.row_stack((e1, e2, e3))
    if distance_metric == 'hop':
        g.add_edges_from(edges)
    elif distance_metric == 'l2':
        dist = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        for e, d in zip(edges, dist):
            g.add_edge(e[0], e[1], weight=d)
    else:
        assert False, 'unknown distance metric'

    node2dist = nx.algorithms.multi_source_dijkstra_path_length(g, idxsites.tolist())
    dist_output = -np.ones(nv)

    for key, val in node2dist.items():
        dist_output[key] = val

    dist_output[dist_output < 0] = fillval
    return dist_output


def mesh_distance_transform(faces, idxsites,
                            vertices=None, distance_metric='hop',
                            fillval=np.inf):
    ''' for each vertex in a mesh, find the shortest distance to one of the sites
    
    Parameters
    -------------------------
    faces
        mesh faces
    idxsites
        indices of vertices to which distances are computed
    vertices
        vertices of the mesh. When distance_metric is 'hop', vertex location is not needed
    distance_metric
        'l2' or 'hop'. 'l2' is the l2 norm, 'hop' is the number of hops from any site to the target vertex
    fillval
        the distance is set to this value if the vertex is not reachable from any site
        
    Returns
    ----------------------
    distance_array
        an array of the length max(faces)+1, which stores the distance of each vertex.
        Nan if the distance is meaningless (e.g., a vertex not contained in any face).
    '''
    import networkx as nx
    from queue import PriorityQueue as PQ

    # create the mesh graph
    g = nx.Graph()
    for i in range(faces.shape[-1] - 1):
        for j in range(i + 1, faces.shape[-1]):
            v1 = faces[:, i]
            v2 = faces[:, j]
            if distance_metric == 'l2':
                w = np.linalg.norm(vertices[v1] - vertices[v2], axis=1)
            elif distance_metric == 'hop':
                w = np.ones(len(v1))
            for k in range(len(v1)):
                g.add_edge(v1[k], v2[k], weight=w[k])

    # run BFS from the chosen vertices
    dist_per_v = np.zeros(np.max(faces) + 1)
    dist_per_v[:] = np.inf
    dist_per_v[idxsites] = 0
    visit_mask = np.zeros(len(dist_per_v), dtype=bool)

    nodes_to_visit = PQ()

    # nodes_to_visit = set(idxsites)
    nodes_next_round = set(idxsites)
    while True:
        while not nodes_to_visit.empty():
            d_node, node = nodes_to_visit.get()
            visit_mask[node] = True

            for nb in g.neighbors(node):
                if visit_mask[nb]:  # skip if already has distance computed
                    continue

                d_new = g.get_edge_data(node, nb)['weight'] + d_node
                if d_new < dist_per_v[nb]:
                    dist_per_v[nb] = d_new

                # add neighbor to next round
                nodes_next_round.add(nb)
        if len(nodes_next_round) == 0:  # nothing to visit next
            break
        else:
            for x in nodes_next_round:
                nodes_to_visit.put((dist_per_v[x], x))
            nodes_next_round = set()

    dist_per_v[~np.isfinite(dist_per_v)] = fillval
    return dist_per_v


def bridge_components(pts, adjmat, knn, use_fast=True):
    ''' connect components of a mesh using knn point-to-point connection.
    A mesh might have several disconnected components, and these components may not
    move together in optimization, because laplacian smoothness will not connect them.
    We can create some arbitrary connection to bind these components so that their positions
    affect each other.
    
    Parameters
    -----------------------------
    pts
        mesh points
    adjmat
        mesh adjacency matrix
    knn
        how many points to connect between components
        
    Return
    ----------------------------
    edges
        edges[i]=(u,v) iff pts[u] should connect to pts[v]
    '''
    import networkx as nx
    import scipy.sparse as sparse
    import scipy.spatial as spatial
    import shortfunc as sf

    if sparse.issparse(adjmat):
        ii, jj = adjmat.nonzero()
    else:
        ii, jj = np.nonzero(adjmat)

    g = nx.Graph()
    g.add_edges_from(np.column_stack((ii, jj)))

    # cluster points by connectivity
    idxpts_per_comp = []
    for subg in nx.connected_components(g):
        idxpts_per_comp.append(np.array(list(subg)))
    n_comp = len(idxpts_per_comp)

    # distance between clusters by nearest points
    kd_per_comp = []
    for idx in idxpts_per_comp:
        kd = spatial.cKDTree(pts[idx])
        kd_per_comp.append(kd)

    distmat = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            if use_fast:
                p_i = pts[idxpts_per_comp[i]]
                p_j = pts[idxpts_per_comp[j]]
                d = np.linalg.norm(p_i.mean(axis=0) - p_j.mean(axis=0))
                distmat[i, j] = d
                distmat[j, i] = d
            else:
                kd_i = kd_per_comp[i]
                p_i = pts[idxpts_per_comp[i]]

                kd_j = kd_per_comp[j]
                p_j = pts[idxpts_per_comp[j]]

                dist_ij, _ = kd_j.query(p_i)
                dist_ji, _ = kd_i.query(p_j)
                d = np.min(np.concatenate((dist_ij, dist_ji)))
                distmat[i, j] = d
                distmat[j, i] = d

    # create MST among clusters
    cluster_graph = nx.Graph()
    for i in range(distmat.shape[0]):
        for j in range(i + 1, distmat.shape[1]):
            cluster_graph.add_edge(i, j, weight=distmat[i, j])
    mst_edges = list(nx.minimum_spanning_edges(cluster_graph))  # each edge is (u,v,weight)
    cluster_edges = [(u, v) for u, v, w in mst_edges]

    # connect the clusters by picking knn points
    final_edges = []
    for u, v in cluster_edges:
        kd_u = kd_per_comp[u]
        idx_u = idxpts_per_comp[u]
        p_u = pts[idx_u]

        kd_v = kd_per_comp[v]
        idx_v = idxpts_per_comp[v]
        p_v = pts[idx_v]

        # distmat_u2v = sparse.lil_matrix((len(p_u), len(p_v)))
        dist_1, idx_1 = kd_u.query(p_v)
        uu = idx_u[idx_1]
        vv = idx_v
        edge_1 = np.column_stack((uu, vv))

        dist_2, idx_2 = kd_v.query(p_u)
        uu = idx_u
        vv = idx_v[idx_2]
        edge_2 = np.column_stack((uu, vv))

        # remove duplicated edges
        dist = np.concatenate((dist_1, dist_2))
        edges = np.row_stack((edge_1, edge_2))
        edges, idxuse = sf.unique_rows(edges)[:2]
        dist = dist[idxuse]

        # rank the edges and select the top-n
        idxsort = np.argsort(dist)
        idx = idxsort[:knn]
        e = edges[idx]
        final_edges.append(e)

        # ii = np.concatenate((idx_u, np.arange(len(p_u))))
        # jj = np.concatenate((np.arange(len(p_v)), idx_v))
        # d = np.concatenate((dist_v2u, dist_u2v))
        # distmat_u2v = sparse.csr_matrix((d, (ii,jj)), shape=(len(p_u), len(p_v)))
        # distmat_u2v[np.arange(len(p_u)), idx] = dist

        # idxu, idxv, dist = sparse.find(distmat)
        # idxsort = np.argsort(dist)
        # idxuse = idxsort[:knn]
        # e = np.column_stack((idxu[idxuse], idxv[idxuse]))
        # final_edges.append(e)
    final_edges = np.row_stack(final_edges)
    return final_edges


def fit_plane(pts3d):
    ''' fit a plane to a set of 3d points
    
    return (p0, normal)
    '''
    p0 = pts3d.mean(axis=0)
    ptsmean = pts3d - p0
    [u, s, v] = np.linalg.svd(ptsmean.T.dot(ptsmean))
    normal = v[-1] / np.linalg.norm(v[-1])
    # normal = normal / np.linalg.norm(normal)
    return (p0, normal)


def get_middle_point(v_list):
    middle_point = np.array([0.0, 0.0, 0.0])
    for vi in v_list:
        middle_point += vi
    middle_point = middle_point / len(v_list)
    return middle_point


# generate an arbitrary ortho frame given a 3d direction
# return (x,y,z), each is a direction, x is the given direction
def make_frame(dir3):
    t = np.array(dir3)
    idxmax = np.argmax(np.abs(t))
    if idxmax == 0:
        t[0], t[1] = t[1], -t[0]
    elif idxmax == 1:
        t[0], t[1] = -t[1], t[0]
    else:
        t[0], t[2] = -t[2], t[0]

    x = np.array(dir3)
    y = np.cross(x, t)
    z = np.cross(x, y)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    return (x, y, z)


def make_nx_graph_by_mesh(faces, vertex_indices=None):
    ''' create networkx graph from mesh.
    
    parameter
    -------------------
    faces
        the nx3 face array
    vertex_indices
        all vertex indices in the graph. If None, use unique(faces)
        
    return
    ------------------
    graph
        a networkx graph object with vertices as nodes and face edges as edges
    '''
    edges = []
    ndim = faces.shape[-1]
    for i in range(ndim - 1):
        for j in range(i + 1, ndim):
            e = faces[:, [i, j]]
            edges.append(e)
    edges = np.row_stack(edges)

    if vertex_indices is None:
        vertex_indices = np.unique(faces)

    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(vertex_indices)
    g.add_edges_from(edges)
    return g


# 2d point-in-mesh test
def point_in_mesh_2(query_pts, vertices, faces):
    ''' test if a 2d point is inside a 2d mesh
    
    Parameters
    ----------------------
    query_pts
        nx2 points, each point will be tested against the mesh
    vertices
        nx2 points, vertices of the mesh
    faces
        nx3 face array
        
    return
    -----------------------
    in_mesh_mask
        mask[i]==True iff query_pts[i] is in the mesh
    idxtri
        idxtri[i]=k iff query_pts[i] is in face[k].
        k=-1 if i-th point is not inside the mesh
    '''
    DIST_THRES = 1e-8
    import meshproc
    query_pts = np.atleast_2d(query_pts)

    assert vertices.shape[-1] == 2, 'vertices must be 2d points'
    assert query_pts.shape[-1] == 2, 'query_pts must be 2d points'

    # extend to 3d
    v_3d = np.column_stack((vertices, np.zeros(len(vertices))))
    pts_3d = np.column_stack((query_pts, np.zeros(len(query_pts))))
    mq = meshproc.MeshQuery()
    mq.set_data(v_3d, faces)
    nnpts, idxtri = mq.closest_points(pts_3d)
    dist = np.linalg.norm(pts_3d - nnpts, axis=1)
    mask = dist < DIST_THRES
    idxtri[~mask] = -1
    return mask, idxtri


# create rotation matrix in 2d
def rotation_matrix_2(angle_rad):
    ''' create 2d rotation matrix by rotation angle relative to x axis.
    
    Parameters
    -----------------------
    angle_rad
        rotation angle by radian
        
    return
    ---------------------
    rotmat
        2x2 rotation matrix. points.dot(rotmat) will rotate all the 2d points
        by angle_rad counter-clockwise
    '''
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    rotmat = np.array([[cos, -sin], [sin, cos]]).T
    return rotmat


# split mesh by face connectivity
def split_mesh(faces):
    ''' split mesh by face connectivity
    
    Parameters
    ----------------------
    faces
        nx3 face array
        
    return
    ----------------------
    idxfacelist
        idxfacelist[i] = indices of faces in the i-th component
    '''
    import networkx as nx
    import shortfunc as sf
    f = np.sort(faces, axis=1)
    edges = np.row_stack((f[:, [0, 1]], f[:, [0, 2]], f[:, [1, 2]]))
    edges = sf.unique_rows(edges)[0]

    g = nx.Graph()
    g.add_edges_from(edges)

    # extract subgraphs
    idxfacelist = []
    for nodes in nx.connected_components(g):
        nodes = list(nodes)
        mask = np.in1d(faces, nodes).reshape(faces.shape).all(axis=1)
        idxface = np.nonzero(mask)[0]
        idxfacelist.append(idxface)
    return idxfacelist


def merge_mesh_overlap_vertices(vertices, faces, dist_thres=1e-5, max_num_overlap=5):
    ''' like merge_mesh_vertices(), but automatically find very close vertices
    to merge.
    
    parameter
    --------------------------
    vertices
        vertices of the mesh
    faces
        nx3 face array of the mesh
    dist_thres
        if two points are closer than this distance, they are considered overlapping
    max_num_overlap
        how many overlapping vertices is allowed for each vertex?
        
    return
    -------------------------
    new_vertices
        new vertices
    new_faces
        new faces
    mappings
        a dict containing mappings of vertices and faces between
        the input mesh and output mesh. Including:
             
        'vertex_old2new': vertex_old2new[i]=j iff vertices[i] is mapped to new_vertices[j]
        
        'face_new2old': face_new2old[i]=j iff new_faces[i] is the same as faces[j]
    '''
    dist, idx = find_closest_points_among_self(vertices, k=max_num_overlap)
    mask = dist < dist_thres
    ii = np.repeat(np.arange(len(vertices)), max_num_overlap).reshape((-1, max_num_overlap))
    jj = idx
    idx_src = ii[mask]
    idx_dst = jj[mask]

    return merge_mesh_vertices(vertices, faces, idx_src=idx_src, idx_target=idx_dst)


# merge mesh vertices
def merge_mesh_vertices(vertices, faces, idx_src, idx_target):
    ''' replace some vertex in a mesh.
    
    Parameters
    --------------------------
    vertices
        vertices of the mesh
    faces
        nx3 face array of the mesh
    idx_src
        indices of the vertices to be merged into other vertices.
        In the end, vertex idx_src[i] will be merged into idx_target[i]
    idx_target
        indices of the vertices what will absorb other vertices.
        In the end, vertex idx_src[i] will be merged into idx_target[i]
        
    Return
    ----------------------------
    new_vertices
        new vertices
    new_faces
        new faces
    mappings
        a dict containing mappings of vertices and faces between
        the input mesh and output mesh. Including:
        
        'src2new': src2new[i]=j iff idx_src[i] is mapped to new_vertices[j]
        
        'target2new': target2new[i]=j iff idx_target[i] is mapped to new_vertices[j]
        
        'vertex_old2new': vertex_old2new[i]=j iff vertices[i] is mapped to new_vertices[j]
        
        'face_new2old': face_new2old[i]=j iff new_faces[i] is the same as faces[j]
    '''
    assert len(idx_src) == len(idx_target), 'number of source and target vertices must be the same'

    # how many vertices do we need to remove?
    uidx_remove = np.unique(np.concatenate((idx_src, idx_target)))  # remove both source and target
    n_remove = len(uidx_remove)
    n_retain = len(vertices) - n_remove

    # establish change of label
    mask_retain = np.ones(len(vertices), dtype=bool)
    mask_retain[idx_src] = False
    mask_retain[idx_target] = False
    label_old2new = np.zeros(len(vertices), dtype=int)
    label_old2new[mask_retain] = np.arange(mask_retain.sum())

    # compute new labels for vertices involved in source and target
    import networkx as nx
    g = nx.Graph()
    x = np.column_stack((idx_src, idx_target))
    g.add_edges_from(x)
    ijs = []
    idx_new_vertex = n_retain  # label of the next new vertex
    idx_nv_in_old = []  # new vertex in old vertex array
    for nodes in nx.connected_components(g):
        n = list(nodes)
        label_old2new[n] = idx_new_vertex
        idx_new_vertex += 1
        idx_nv_in_old.append(n[0])

    # new label and position for new vertex
    newv_label = np.arange(len(idx_nv_in_old)) + n_retain
    newv_position = vertices[idx_nv_in_old]

    mappings = {}
    mappings['src2new'] = label_old2new[idx_src]
    mappings['target2new'] = label_old2new[idx_target]
    mappings['vertex_old2new'] = label_old2new

    # create new faces by changing labels
    new_faces = label_old2new[faces]

    # create new vertices
    vnew = np.row_stack((vertices[mask_retain], newv_position))

    # find invalid faces and remove them
    f1, f2, f3 = new_faces.T
    mask_invalid = (f1 == f2) | (f1 == f3) | (f2 == f3)
    idx = np.nonzero(~mask_invalid)[0]

    new_faces = new_faces[idx]
    idxface_new2old = idx
    mappings['face_new2old'] = idxface_new2old

    return vnew, new_faces, mappings


def find_closest_points_among_self(pts, idx_query_pts=None, k=1):
    ''' this function finds closest points in a point set using some of the 
    points in the same point set as query points.
    
    parameter
    ---------------------------
    pts
        the whole point set
    idx_query_pts
        indices to pts, specifying which points are used as query points.
        For each point in pts[idx_query_pts], we find the closest points in
        pts such that it will not find itself back. If none, all points in
        pts are used as query points
    k
        find K-nearest points
        
    return
    -----------------------------
    dist
        nxk matrix, distance, where dist[i,j] is the distance from pts[idx_query_pts][i] to
        its j-th closest point
    idx
        nxk matrix, indices of closest points, pts[idx[i,j]] is the j-th closest point 
        for pts[idx_query_pts][i].
    '''
    if idx_query_pts is None:
        idx_query_pts = np.arange(len(pts))
    else:
        idx_query_pts = np.array(idx_query_pts).flatten()

    import scipy.spatial as spatial
    kd = spatial.cKDTree(pts)
    query_pts = pts[idx_query_pts]
    distmat, idxmat = kd.query(query_pts, k=k + 1)

    # mask_match_self[i,j] == True iff
    # idxmat[i,j] = idx_query_pts[i], which means the query point matches itself
    mask_match_self = np.zeros(idxmat.shape, dtype=bool)
    for i in range(idxmat.shape[-1]):
        m = idxmat[:, i] == idx_query_pts
        mask_match_self[:, i] = m
    m = mask_match_self.any(axis=1)

    # if a row has no self-match, then we set the closest point as self match
    mask_match_self[~m, 0] = True

    # get rid of self match in distmat and idxmat
    shape = (distmat.shape[0], distmat.shape[1] - 1)
    dist = distmat[~mask_match_self].reshape(shape)
    idx = idxmat[~mask_match_self].reshape(shape)

    return dist, idx


def compute_vertex_normals(vertices, faces):
    ''' compute vertex normals for a mesh
    
    return
    -------------------------
    vertex_normal
        vertex_normal[i] is the normal of vertices[i]
    '''
    import trimesh
    obj = trimesh.Trimesh(vertices, faces, process=False)
    vn = np.array(obj.vertex_normals)
    return vn


def smooth_mesh_by_laplacian(vertices, faces, w_smooth=1.0):
    ''' smooth mesh by setting laplacian=0
    
    return
    --------------------
    vertices_smooth
        smoothed vertices
    '''
    import scipy.sparse as sparse
    from . import shortfunc as sf

    n_vert = len(vertices)
    adjmat = adjmat_from_mesh(len(vertices), faces)
    lpmat = laplacian_from_adjmat(adjmat, normalize=True)
    A_fix = sparse.eye(n_vert)
    b_fix = vertices

    A_smooth = lpmat
    b_smooth = np.zeros(vertices.shape)

    A = sparse.vstack((A_fix, A_smooth * w_smooth))
    b = np.row_stack((b_fix, b_smooth * w_smooth))

    sol = sf.mldivide(A, b)
    return sol


def find_most_distant_pair(pts):
    ''' find the pair of points that has the longest distance among all point pairs
    
    parameter
    ------------------
    pts
        nxd points
        
    return
    -------------------
    idx
        [p1,p2], indices of the pair of points that has the longest distance
    distance
        the distance between the pair of points
    '''
    idx_1 = None
    idx_2 = None
    maxdist = 0

    for i in range(len(pts)):
        dist = np.linalg.norm(pts[i:] - pts[i], axis=1)
        idxmax = np.argmax(dist)
        if dist[idxmax] > maxdist:
            maxdist = dist[idxmax]
            idx_1 = i
            idx_2 = idxmax + i

    return (idx_1, idx_2), maxdist


def align_mesh_by_distant_point_pair_v3(v_src, v_dst, n_cand=3, transform_type='projective', num_vertex_test=100):
    ''' align two models with the same number of vertices,
    but in different vertex order.The two models differ 
    by a uniform similarity transform with possible reflection.
    
    parameter
    ---------------------------
    v_src
        source model vertices
    v_dst
        destination model vertices
    n_pool, n_pick
        we will find n_pick correspondences between src and dst, by first
        sorting the vertices according to some feature, and test all combinations
        that contain n_pick vertices among the first n_pool vertices
    num_vertex_test
        when testing fit, only use this many vertices. set to None or Inf
        to use all vertices
        
    return
    ----------------------------
    transmat
        4x4 transformation matrix that transforms src into dst
    '''
    from . import shortfunc as sf
    import itertools
    import scipy.spatial as spatial

    if num_vertex_test is None:
        num_vertex_test = np.inf

    num_vertex_test = np.min([num_vertex_test, len(v_src)])

    # find farthest point pairs among src and dst
    idx, fardist_src = find_most_distant_pair(v_src)
    idx_src_1, idx_src_2 = idx
    center_src = (v_src[idx_src_1] + v_src[idx_src_2]) / 2
    dist_src = np.linalg.norm(v_src - center_src, axis=1)

    idx, fardist_dst = find_most_distant_pair(v_dst)
    idx_dst_1, idx_dst_2 = idx
    center_dst = (v_dst[idx_dst_1] + v_dst[idx_dst_2]) / 2
    dist_dst = np.linalg.norm(v_dst - center_dst, axis=1)

    # normalize scale
    scale = fardist_dst / fardist_src
    dist_src *= scale

    # pick several points from src for corresondence
    n_pick = 4
    idxpick = np.linspace(0, len(v_src) - 1, n_pick).astype(int)
    p_src = v_src[idxpick]

    # find knn in dst using center dist as feature
    feature_dst = dist_dst.reshape((-1, 1))
    feature_src = dist_src.reshape((-1, 1))
    kd_feature_dst = spatial.cKDTree(feature_dst)
    dist, idx_dst_cand = kd_feature_dst.query(feature_src[idxpick], n_cand)

    best_transmat = None
    best_distance = np.inf
    kd_dst = spatial.cKDTree(v_dst)
    for idx in itertools.product(*idx_dst_cand.tolist()):
        idx = np.array(idx)
        p_dst = v_dst[idx]
        if transform_type == 'projective':
            tmat = sf.find_transform_projective(p_src, p_dst)
        elif transform_type == 'similarity':
            tmat = sf.find_transform_similarity(p_src, p_dst)
        else:
            assert False, 'unknown transformation type'

        v_src_transform = sf.transform_points(v_src, tmat)
        dist, _ = kd_dst.query(v_src_transform[:num_vertex_test])
        dist = dist.sum()
        print(dist)
        if dist < best_distance:
            best_distance = dist
            best_transmat = tmat

    return best_transmat


def align_mesh_by_distant_point_pair_v2(v_src, v_dst, n_pool=10, n_pick=5, transform_type='projective',
                                        num_vertex_test=100):
    ''' align two models with the same number of vertices,
    but in different vertex order.The two models differ 
    by a uniform similarity transform with possible reflection.
    
    parameter
    ---------------------------
    v_src
        source model vertices
    v_dst
        destination model vertices
    n_pool, n_pick
        we will find n_pick correspondences between src and dst, by first
        sorting the vertices according to some feature, and test all combinations
        that contain n_pick vertices among the first n_pool vertices
    num_vertex_test
        when testing fit, only use this many vertices. set to None or Inf
        to use all vertices
        
    return
    ----------------------------
    transmat
        4x4 transformation matrix that transforms src into dst
    '''
    from . import shortfunc as sf
    import itertools
    import scipy.spatial as spatial

    if num_vertex_test is None:
        num_vertex_test = np.inf

    num_vertex_test = np.min([num_vertex_test, len(v_src)])

    # find farthest point pairs among src and dst
    idx, dist_src = find_most_distant_pair(v_src)
    idx_src_1, idx_src_2 = idx
    center_src = (v_src[idx_src_1] + v_src[idx_src_2]) / 2
    dist_src = np.linalg.norm(v_src - center_src, axis=1)
    idxsort_src = np.argsort(dist_src)

    idx, dist_dst = find_most_distant_pair(v_dst)
    idx_dst_1, idx_dst_2 = idx
    center_dst = (v_dst[idx_dst_1] + v_dst[idx_dst_2]) / 2
    dist_dst = np.linalg.norm(v_dst - center_dst, axis=1)
    idxsort_dst = np.argsort(dist_dst)

    idx_dst_cand = idxsort_dst[:n_pool].tolist()
    p_src = v_src[idxsort_src[:n_pick]]

    best_transmat = None
    best_distance = np.inf

    kd_dst = spatial.cKDTree(v_dst)
    for x in itertools.combinations(idx_dst_cand, n_pick):
        idx = np.array(x)
        p_dst = v_dst[idx]
        if transform_type == 'projective':
            tmat = sf.find_transform_projective(p_src, p_dst)
        elif transform_type == 'similarity':
            tmat = sf.find_transform_similarity(p_src, p_dst)
        else:
            assert False, 'unknown transformation type'

        v_src_transform = sf.transform_points(v_src, tmat)
        dist, _ = kd_dst.query(v_src_transform[:num_vertex_test])
        dist = dist.sum()
        if dist < best_distance:
            best_distance = dist
            best_transmat = tmat
    return best_transmat


def align_mesh_by_distant_point_pair(v_src, v_dst, num_vertex_sample=7):
    ''' align two models with the same number of vertices,
    but in different vertex order.The two models differ 
    by a uniform similarity transform with possible reflection.
    
    parameter
    ---------------------------
    v_src
        source model vertices
    v_dst
        destination model vertices
    num_vertex_sample
        number of vertices used for alignment
        
    return
    ----------------------------
    transmat
        4x4 transformation matrix that transforms src into dst
    '''
    from . import shortfunc as sf

    # find farthest point pairs among src and dst
    idx, dist_src = find_most_distant_pair(v_src)
    idx_src_1, idx_src_2 = idx
    center_src = (v_src[idx_src_1] + v_src[idx_src_2]) / 2
    dist_src = np.linalg.norm(v_src - center_src, axis=1)
    idxsort_src = np.argsort(dist_src)

    idx, dist_dst = find_most_distant_pair(v_dst)
    idx_dst_1, idx_dst_2 = idx
    center_dst = (v_dst[idx_dst_1] + v_dst[idx_dst_2]) / 2
    dist_dst = np.linalg.norm(v_dst - center_dst, axis=1)
    idxsort_dst = np.argsort(dist_dst)

    vidx_src = idxsort_src[:num_vertex_sample]
    vidx_dst = idxsort_dst[:num_vertex_sample]

    # align them
    tmat = sf.find_transform_projective(v_src[vidx_src], v_dst[vidx_dst])
    return tmat


def map_vertices_by_laplacian(v_src, f_src, v_dst, f_dst):
    ''' align two models with the same number of vertices,
    but in different vertex order.The two models differ 
    by a uniform similarity transform.
    
    parameter
    ---------------------------
    v_src, f_src
        source model vertices and faces
    v_dst, f_dst
        destination model vertices and faces
        
    return
    ----------------------------
    idx_dst
        indices of v_dst that correspond to v_src, such that 
        v_dst[idx_dst][i] is the correspondence point of v_src[i]
    '''
    adjmat = adjmat_from_mesh(len(v_src), f_src)
    lpmat_src = laplacian_from_adjmat(adjmat)
    lpval_src = lpmat_src.dot(v_src)
    slen_src = np.linalg.norm(lpval_src, axis=1).sum()

    adjmat = adjmat_from_mesh(len(v_dst), f_dst)
    lpmat_dst = laplacian_from_adjmat(adjmat)
    lpval_dst = lpmat_dst.dot(v_dst)
    slen_dst = np.linalg.norm(lpval_dst, axis=1).sum()

    scale_src2dst = slen_dst / slen_src
    lpval_src_scale = lpval_src * scale_src2dst

    import scipy.spatial as spatial
    kd = spatial.cKDTree(lpval_dst)
    _, idx = kd.query(lpval_src_scale)
    return idx

    # tmat_src2dst = sf.find_transform_similarity(v_src, v_dst[idx])

    # v_obj_transform = sf.transform_points(v_obj, tmat_obj2cof)
    # import myplot.vtkplot as vp
    # vp.trisurf(v_obj_transform, model_obj.faces, color3f=(0,0,0))
    # vp.trisurf(v_cof, model_cof.faces, color3f=(1,0,0))
    # vp.show()


def make_rays_in_cone_3(central_direction, theta, n_theta, n_alpha, return_parameter=False):
    ''' given a 3d ray A along the direction central_direction, generate some rays B
    such that the angle between A and B is less than theta.
    
    parameter
    --------------------------
    central_direction
        the direction of the central ray
    theta
        max angle between central ray and generated ray, in radian
    n_theta
        number of slices in theta
    n_alpha
        number of slices on the circle around the ray
    return_parameter
        return the (theta,alpha) for each generated ray
        
    return
    --------------------------
    ray_dirs
        nx3 matrix, each row is a ray direction
    theta_alpha
        nx2 matrix, returned if return_parameter==True.
        each row is the (theta, alpha) parameter of the generated ray, 
        such that ray_dirs[i] = (sin(theta)cos(alpha), sin(theta)sin(alpha), cos(theta))
    '''
    theta_list = np.linspace(0, theta, n_theta + 1)[1:]
    alpha_list = np.linspace(0, np.pi * 2, n_alpha + 1)[1:]
    tt, aa = np.meshgrid(theta_list, alpha_list)

    # the generated rays are pointing towards z axis
    x = np.sin(tt) * np.cos(aa)
    y = np.sin(tt) * np.sin(aa)
    z = np.cos(tt)
    rays = np.column_stack((x.flat, y.flat, z.flat))
    ttaa = np.column_stack((tt.flat, aa.flat))

    from shortfunc import make_frame
    rotmat = make_frame(central_direction)
    rays = rays.dot(rotmat)  # the generated rays are pointing towards z axis

    if return_parameter:
        return rays, ttaa
    else:
        return rays


# move points by Laplacian deformation
def deform_by_laplacian(pts, adjmat, idxmove, ptsmove,
                        w_move=1.0, w_laplacian=1.0,
                        smooth_by_lp_zero=False):
    ''' move some points of a graph to destination, and other points are moved along
    by laplacian smoothing.

    parameter
    ----------------------
    pts
        all points of the graph
    adjmat
        adjacency matrix of the graph
    idxmove
        indices of the graph vertices to be moved
    ptsmove
        destination of the moved vertices
    w_move
        weight of pts[idxmove] being close to ptsmove.
        If w_move is a vector same length ahs len(idxmove), then it
        is the weight for each moved vertex.
    w_laplacian
        weight of smoothing
    smooth_by_lp_zero
        if True, the smoothing is done by setting laplacian coordinates to 0.
        if False, laplacian coordinates of the original graph is preserved

    return
    ----------------------
    pts_final
        the moved graph vertices
    '''
    # lp matrix
    from shortfunc import adjmat_to_laplacian, mldivide
    import scipy.sparse as sparse

    if np.isscalar(w_move):
        w_move = np.ones(len(idxmove)) * w_move

    if np.isscalar(w_laplacian):
        w_laplacian = np.ones(len(pts)) * w_laplacian

    assert len(w_move) == len(idxmove)

    adjmat = sparse.csc_matrix(adjmat)
    L = adjmat_to_laplacian(adjmat)

    # lp coordinate
    lpc = L.dot(pts)

    # fixed point matrix
    # ii = np.arange(len(idxmove))
    # jj = idxmove
    # E = sparse.lil_matrix((len(idxmove),len(pts)))
    # E[ii,jj] = 1
    E = sparse.eye(len(pts), format='csr')[idxmove]

    # formulate
    A = sparse.vstack((L.multiply(w_laplacian.reshape((-1, 1))), E.multiply(w_move.reshape((-1, 1)))))
    if smooth_by_lp_zero:
        bs = np.row_stack((np.zeros_like(lpc), ptsmove * w_move.reshape((-1, 1))))
    else:
        bs = np.row_stack((lpc * w_laplacian.reshape((-1, 1)), ptsmove * w_move.reshape((-1, 1))))
    sol = mldivide(A, bs)

    # solve
    # AtA = A.transpose().dot(A)
    # Atb = A.transpose().dot(bs)
    # lu = slg.splu(AtA)
    # sol = lu.solve(Atb)

    return sol


def barycentric_to_real_coordinate(bcpts, idxface, vertices, faces):
    ''' convert barycentric coordinate on a mesh to true coordinates
    
    parameter
    -----------------------
    bcpts
        NxD barycentric coordinates
    idxface
        idxface[i] is the face that contains bcpts[i]
    vertices
        the mesh vertices
    faces
        the mesh faces
        
    return
    --------------------
    pts
        the true coordinates on the mesh
    '''
    bcpts = np.atleast_2d(bcpts)
    outpts = np.zeros((len(bcpts), vertices.shape[-1]))
    for i in range(bcpts.shape[-1]):
        b = bcpts[:, i]
        idxv = faces[idxface, i]
        outpts += vertices[idxv] * b.reshape((-1, 1))
    return outpts


def merge_models(model_data_list):
    ''' combine a list of ModelData objects into one single ModelData object
    
    parameter
    --------------------
    model_data_list
        a list of ModelData objects
        
    return
    -------------------
    model_data
        a single ModelData object containing all contents in model_data_list
    '''
    from modeldata.modeldata import ModelData
    import trimesh

    if False:
        model = ModelData()

    # merge vertices and faces
    num_vert = 0
    v_new = []
    f_new = []
    for model in model_data_list:
        v_new.append(model.vertices)
        f_new.append(model.faces + num_vert)
        num_vert += len(model.vertices)
    v_new = np.row_stack(v_new)
    f_new = np.row_stack(f_new)

    # merge texture coordinates
    has_tex_coord = [x.texcoord_uv is not None for x in model_data_list]
    has_tex_coord = np.any(has_tex_coord)
    if has_tex_coord:
        num_vert = 0
        vt_new = []
        ft_new = []
        for model in model_data_list:
            vt = model.texcoord_uv
            ft = model.texcoord_faces
            if vt is None:
                vt = np.zeros_like(model.vertices)
                ft = model.faces
            vt_new.append(vt)
            ft_new.append(ft + num_vert)
            num_vert += len(vt)
        vt_new = np.row_stack(vt_new)
        ft_new = np.row_stack(ft_new)
    else:
        vt_new = None
        ft_new = None

    # merge normals
    vn_new = [model.get_vertex_normals() for model in model_data_list]
    vn_new = np.row_stack(vn_new)

    fn_new = [model.get_face_normals() for model in model_data_list]
    fn_new = np.row_stack(fn_new)

    # get texture image
    teximg = None
    for model in model_data_list:
        if model.texture_image is not None:
            teximg = model.texture_image
            break

    # create model
    model = ModelData(vertex_local=v_new, faces=f_new,
                      texcoord_uvw=vt_new, texcoord_faces=ft_new,
                      texture_image=teximg)
    model._vertex_normals = vn_new
    model._face_normals = fn_new
    return model


def dist2poly2d(pts2d_test, pts2d_poly):
    '''
    point distance to polygon
    compute distance from pts2d_test to a closed polygon defined by pts2d_poly
    
    return dist, where dist[i] is the distance from pts2d_test[i] to the polygon
    '''
    import cv2
    distlist = []
    pts2d_poly = pts2d_poly.astype('float32')
    for p in pts2d_test:
        d = cv2.pointPolygonTest(pts2d_poly, tuple(p), True)
        distlist.append(d)
    distlist = np.abs(np.array(distlist))
    return distlist


def fit_line(pts, line_dir=None):
    ''' fit a line to a list of points
    
    parameter
    -------------------
    line_dir
        specified line direction, it will be determined automatically if not provided
    
    return
    -------------------
    p0
        a point on the plane
    line_direction
        the direction of the fitted line    
    '''
    if line_dir is None:
        pts = np.array(pts)
        ptsmean = pts - pts.mean(axis=0)
        [u, s, v] = np.linalg.svd(ptsmean.T.dot(ptsmean))
        line_dir = v[0]
    line_dir = np.array(line_dir).flatten()
    p0 = pts.mean(axis=0)

    return p0, line_dir


def fit_rectangle(pts2d, outdict=None):
    '''
    fit a rectangle to a list of points
    return ((center_x,center_y), (width, height), angle(rad)), angle is CCW rotation from x+ axis
    
    outdict['distlist'] = distance from pts2d[i] to the nearest poin
    '''
    import cv2
    pts = pts2d.astype('float32')
    center, size, angle = cv2.minAreaRect(pts)
    angle = np.deg2rad(angle)

    if outdict is not None:
        wdir = np.array([np.cos(angle), np.sin(angle)])
        hdir = np.array([-np.sin(angle), np.cos(angle)])

        dw = wdir * size[0]
        dh = hdir * size[1]
        c = np.array(center)
        p0 = c + dw / 2 + dh / 2
        p1 = p0 - dh
        p2 = p1 - dw
        p3 = p2 + dh
        ptsrect = np.array([p0, p1, p2, p3, p0])
        distlist = dist2poly2d(pts2d, ptsrect)
        outdict['distlist'] = distlist

    return (center, size, angle)


def find_closest_point_with_barycentric(pts, v_mesh, f_mesh, pgmesh=None, outdict=None):
    import sys
    sys.path.insert(0, "D:/mproject/face-reconstruct/")
    ''' for each point on pts, find the closest point on a given mesh,
    return the closest points and the barycentric coordinates. The mesh and query
    points can be 2d or 3d, if they are 2d, their z coordinates are set to 0.
    
    parameter
    ----------------------
    pts
        the query points
    v_mesh, f_mesh
        vertices and faces of the mesh
    pgmesh
        a pygeom.TriangularMesh object representing the mesh.
        If this is given, v_mesh and f_mesh are ignored.
        Otherwise a pygeom.TriangularMesh will be constructed automatically
    outdict
        a dict for additional output, with keys:
        
        'pgmesh' = the pygeom.TriangularMesh of the mesh
        
    return
    ----------------------
    nnpts
        nearest point for each point in pts
    idxtri
        indices of the triangles that contain nnpts.
        the triangle idxtri[i] contains nnpts[i]
    bcpts
        barycentric coordinates of nnpts
    '''
    import pygeom
    if False:
        pgmesh = pygeom.TriangularMesh()
    if outdict is None:
        outdict = {}

    if pgmesh is None:
        if v_mesh.shape[-1] == 2:
            v_mesh = np.column_stack((v_mesh, np.zeros(len(v_mesh))))
        pgmesh = pygeom.TriangularMesh()
        v_mesh = np.ascontiguousarray(v_mesh, dtype=np.float64)
        f_mesh = np.ascontiguousarray(f_mesh, dtype=np.int)
        print v_mesh.flags['C_CONTIGUOUS'], f_mesh.flags['C_CONTIGUOUS']
        pgmesh.set_data(v_mesh, f_mesh)
    else:
        v_mesh = pgmesh.get_vertices()
        f_mesh = pgmesh.get_faces()

    outdict['pgmesh'] = pgmesh
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    nnpts, idxtri, bcpts = pgmesh.find_closest_points(pts)

    # check for nan's, replace nan barycentric coordinates with
    # the center of the triangle in question
    mask_nan = np.any(np.isnan(bcpts), axis=1)
    bcpts[mask_nan] = 1.0 / 3

    # recompute nnpts by barycentric
    nnpts_new = barycentric_to_real_coordinate(bcpts, idxtri, v_mesh, f_mesh)

    return nnpts_new, idxtri, bcpts


class PlaneTransform(object):
    ''' transform 3d points on a plane to and from 2d
    '''

    def __init__(self, p0, normal, xdir=None, ydir=None):
        ''' initialize the plane transform with a plane
        
        a frame will be created using p0 as origin, [xdir,ydir,normal] as x,y,z axis.
        If xdir and ydir are not given, they are computed automatically.
        
        xdir, ydir = directions of the frame that may be specified by user. You can only specify one of them
        '''
        from .shortfunc import make_frame
        p0 = np.array(p0).flatten()
        normal = np.array(normal).flatten()

        assert xdir is None or ydir is None, 'You cannot specify xdir and ydir at the same time'
        if xdir is not None:
            ydir = np.cross(normal, xdir)
        if ydir is not None:
            xdir = np.cross(ydir, normal)
        if xdir is None and ydir is None:
            xdir, ydir, _ = make_frame(normal)

        normal /= np.linalg.norm(normal)
        xdir = np.array(xdir) / np.linalg.norm(xdir)
        ydir = np.array(ydir) / np.linalg.norm(ydir)

        transmat = np.eye(4)
        transmat[:, :-1] = np.array([xdir, ydir, normal, p0])
        self.transmat = transmat

    def to_2d(self, pts3d):
        from .shortfunc import mrdivide
        pts3d = np.atleast_2d(pts3d)
        pts = np.column_stack((pts3d, np.ones(len(pts3d))))
        pts = mrdivide(pts, self.transmat)
        pts = pts[:, :2] / pts[:, -1:]
        return pts

    def to_3d(self, pts2d):
        pts2d = np.atleast_2d(pts2d)
        pts = np.column_stack((pts2d, np.zeros(len(pts2d)), np.ones(len(pts2d))))
        pts = pts.dot(self.transmat)[:, :-1]
        return pts

    def to_dir_3d(self, dir2d):
        from .shortfunc import transform_vectors
        dir2d = np.atleast_2d(dir2d)
        dirs = np.column_stack((dir2d, np.zeros(len(dir2d))))
        dirs = transform_vectors(dirs, self.transmat)
        return dirs

    @property
    def p0(self):
        return self.transmat[-1, :-1]

    @property
    def normal(self):
        return self.transmat[2, :-1]
