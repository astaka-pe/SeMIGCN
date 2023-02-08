from turtle import pd
import numpy as np
import torch
from functools import reduce
from collections import Counter
import scipy as sp
import heapq
import copy
from sklearn.preprocessing import normalize

OPTIM_VALENCE = 6
VALENCE_WEIGHT = 1

class Mesh:
    def __init__(self, path, manifold=True, build_mat=True, build_code=False):
        self.path = path
        self.vs, self.vc, self.faces = self.fill_from_file(path)
        self.compute_face_normals()
        self.compute_face_center()
        self.device = 'cpu'
        self.simp = False

        if manifold:
            self.build_gemm() #self.edges, self.ve
            self.compute_vert_normals()
            self.build_v2v()
            self.build_vf()
            self.vs_code = None
            if build_mat:
                self.build_mesh_lap()
            if build_code:
                self.vs_code = self.eigen_decomposition(self.lapmat, k=512)
                self.fc_code = self.eigen_decomposition(self.f_lapmat, k=100)

    def fill_from_file(self, path):
        vs, faces, vc = [], [], []
        f = open(path)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
                if len(splitted_line) == 7: # colored mesh
                    vc.append([float(v) for v in splitted_line[4:7]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        vc = np.asarray(vc)
        faces = np.asarray(faces, dtype=int)

        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, vc, faces

    def build_gemm(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count
        # lots of DS for loss
        """
        self.nvs, self.nvsi, self.nvsin, self.ve_in = [], [], [], []
        for i, e in enumerate(self.ve):
            self.nvs.append(len(e))
            self.nvsi += len(e) * [i]
            self.nvsin += list(range(len(e)))
            self.ve_in += e
        self.vei = reduce(lambda a, b: a + b, self.vei, [])
        self.vei = torch.from_numpy(np.array(self.vei).ravel()).to(self.device).long()
        self.nvsi = torch.from_numpy(np.array(self.nvsi).ravel()).to(self.device).long()
        self.nvsin = torch.from_numpy(np.array(self.nvsin).ravel()).to(self.device).long()
        self.ve_in = torch.from_numpy(np.array(self.ve_in).ravel()).to(self.device).long()

        self.max_nvs = max(self.nvs)
        self.nvs = torch.Tensor(self.nvs).to(self.device).float()
        self.edge2key = edge2key
        """

    def compute_face_normals(self):
        face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]])
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
        face_areas = 0.5 * np.sqrt((face_normals**2).sum(axis=1))
        face_normals /= norm
        self.fn, self.fa = face_normals, face_areas

    def compute_vert_normals(self):
        vert_normals = np.zeros((3, len(self.vs)))
        face_normals = self.fn
        faces = self.faces

        nv = len(self.vs)
        nf = len(faces)
        mat_rows = faces.reshape(-1)
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)
        mat_vals = np.ones(len(mat_rows))
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))
        vert_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)
        vert_normals = normalize(vert_normals, norm='l2', axis=1)
        self.vn = vert_normals
    
    def compute_face_center(self):
        faces = self.faces
        vs = self.vs
        self.fc = np.sum(vs[faces], 1) / 3.0
    
    def compute_fn_sphere(self):
        fn = self.fn
        u = (np.arctan2(fn[:, 1], fn[:, 0]) + np.pi) / (2.0 * np.pi)
        v = np.arctan2(np.sqrt(fn[:, 0]**2 + fn[:, 1]**2), fn[:, 2]) / np.pi
        self.fn_uv = np.stack([u, v]).T
    
    def build_uni_lap(self):
        """compute uniform laplacian matrix"""
        vs = torch.tensor(self.vs.T, dtype=torch.float)
        edges = self.edges
        ve = self.ve

        sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
        sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

        num_verts = vs.size(1)
        mat_rows = [np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)]
        mat_rows = np.concatenate(mat_rows)
        mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]
        mat_cols = np.concatenate(mat_cols)

        mat_rows = torch.from_numpy(mat_rows).long()
        mat_cols = torch.from_numpy(mat_cols).long()
        mat_vals = torch.ones_like(mat_rows).float() * -1.0
        neig_mat = torch.sparse.FloatTensor(torch.stack([mat_rows, mat_cols], dim=0),
                                            mat_vals,
                                            size=torch.Size([num_verts, num_verts]))
        vs = vs.T

        sum_count = torch.sparse.mm(neig_mat, torch.ones((num_verts, 1)).type_as(vs))
        mat_rows_ident = np.array([i for i in range(num_verts)])
        mat_cols_ident = np.array([i for i in range(num_verts)])
        mat_ident = np.array([-s for s in sum_count[:, 0]])
        mat_rows_ident = torch.from_numpy(mat_rows_ident).long()
        mat_cols_ident = torch.from_numpy(mat_cols_ident).long()
        mat_ident = torch.from_numpy(mat_ident).long()
        mat_rows = torch.cat([mat_rows, mat_rows_ident])
        mat_cols = torch.cat([mat_cols, mat_cols_ident])
        mat_vals = torch.cat([mat_vals, mat_ident])

        self.lapmat = torch.sparse.FloatTensor(torch.stack([mat_rows, mat_cols], dim=0),
                                            mat_vals,
                                            size=torch.Size([num_verts, num_verts]))
    
    def build_vf(self):
        vf = [set() for _ in range(len(self.vs))]
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf

        """ build vertex-to-face sparse matrix """
        v2f_inds = [[] for _ in range(2)]
        v2f_vals = []
        v2f_areas = [[] for _ in range(len(self.vs))]
        for i in range(len(vf)):
            v2f_inds[1] += list(vf[i])
            v2f_inds[0] += [i] * len(vf[i])
            v2f_vals += (self.fc[list(vf[i])] - self.vs[i].reshape(1, -1)).tolist()
            v2f_areas[i] = np.sum(self.fa[list(vf[i])])
        self.v2f_list = [v2f_inds, v2f_vals, v2f_areas]
        
        v2f_inds = torch.tensor(v2f_inds).long()
        v2f_vals = torch.ones(v2f_inds.shape[1]).float()
        self.v2f_mat = torch.sparse.FloatTensor(v2f_inds, v2f_vals, size=torch.Size([len(self.vs), len(self.faces)]))
        self.f2v_mat = torch.sparse.FloatTensor(torch.stack([v2f_inds[1], v2f_inds[0]], dim=0), v2f_vals, size=torch.Size([len(self.faces), len(self.vs)]))

        """ build face-to-face (1ring) matrix """        
        f_edges = np.array([[i] * 3 for i in range(len(self.faces))])
        f2f = [[] for _ in range(len(self.faces))]
        self.f_edges = [[] for _ in range(2)]
        for i, f in enumerate(self.faces):
            all_neig = list(vf[f[0]]) + list(vf[f[1]]) + list(vf[f[2]])
            one_neig = np.array(list(Counter(all_neig).values())) == 2
            f2f_i = np.array(list(Counter(all_neig).keys()))[one_neig].tolist()
            self.f_edges[0] += len(f2f_i) * [i]
            self.f_edges[1] += f2f_i
            f2f[i] = f2f_i + (3 - len(f2f_i)) * [-1]

        self.f2f = np.array(f2f)
        self.f_edges = np.array(self.f_edges)
        edge_index = torch.tensor(self.edges.T, dtype=torch.long)
        self.edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
        self.face_index = torch.from_numpy(self.f_edges)

        # TODO: change this to correspond to non-watertight mesh
        f2f_inds = torch.from_numpy(self.f_edges).long()
        f2f_vals = -1.0 * torch.ones(f2f_inds.shape[1]).float()
        f2f_mat = torch.sparse.FloatTensor(f2f_inds, f2f_vals, size=torch.Size([len(self.faces), len(self.faces)]))
        f_eyes_inds = torch.arange(len(self.faces)).long().repeat(2, 1)
        f_dims = torch.ones(len(self.faces)).float() * 3.0  # TODO: change here
        f_eyes = torch.sparse.FloatTensor(f_eyes_inds, f_dims, size=torch.Size([len(self.faces), len(self.faces)]))
        self.f_lapmat = f2f_mat + f_eyes
        
        """ build face-to-face (2ring) sparse matrix 
        self.f2f = np.array(f2f)
        f2ring = self.f2f[self.f2f].reshape(-1, 9)
        self.f2ring = [set(f) for f in f2ring]
        self.f2ring = [list(self.f2ring[i] | set(f)) for i, f in enumerate(self.f2f)]
        
        self.f_edges = np.concatenate((self.f2f.reshape(1, -1), f_edges.reshape(1, -1)), 0)
        mat_inds = torch.from_numpy(self.f_edges).long()
        #mat_vals = torch.ones(mat_inds.shape[1]).float()
        mat_vals = torch.from_numpy(self.fa[self.f_edges[0]]).float()
        self.f2f_mat = torch.sparse.FloatTensor(mat_inds, mat_vals, size=torch.Size([len(self.faces), len(self.faces)]))
        """
    def build_v2v(self):
        v2v = [[] for _ in range(len(self.vs))]
        for i, e in enumerate(self.edges):
            v2v[e[0]].append(e[1])
            v2v[e[1]].append(e[0])
        self.v2v = v2v

        """ compute adjacent matrix """
        edges = self.edges
        v2v_inds = edges.T
        v2v_inds = torch.from_numpy(np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1)).long()
        v2v_vals = torch.ones(v2v_inds.shape[1]).float()
        self.Adj = torch.sparse.FloatTensor(v2v_inds, v2v_vals, size=torch.Size([len(self.vs), len(self.vs)]))
        self.v_dims = torch.sum(self.Adj.to_dense(), axis=1)
        D_inds = torch.stack([torch.arange(len(self.vs)), torch.arange(len(self.vs))], dim=0).long()
        D_vals = 1 / self.v_dims
        self.Diag = torch.sparse.FloatTensor(D_inds, D_vals, size=torch.Size([len(self.vs), len(self.vs)]))
        I = torch.eye(len(self.vs))
        self.AdjI = (I + self.Adj).to_sparse()
        Lap = I - torch.sparse.mm(self.Diag, self.Adj.to_dense())
        self.Lap = Lap.to_sparse()    

    def build_adj_mat(self):
        edges = self.edges
        v2v_inds = edges.T
        v2v_inds = torch.from_numpy(np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1)).long()
        v2v_vals = torch.ones(v2v_inds.shape[1]).float()
        self.Adj = torch.sparse.FloatTensor(v2v_inds, v2v_vals, size=torch.Size([len(self.vs), len(self.vs)]))
        self.v_dims = torch.sum(self.Adj.to_dense(), axis=1)
        D_inds = torch.stack([torch.arange(len(self.vs)), torch.arange(len(self.vs))], dim=0).long()
        D_vals = 1 / (torch.pow(self.v_dims, 0.5) + 1.0e-12)
        self.D_minus_half = torch.sparse.FloatTensor(D_inds, D_vals, size=torch.Size([len(self.vs), len(self.vs)]))

    def build_mesh_lap(self):
        self.build_adj_mat()

        vs = self.vs
        edges = self.edges
        faces = self.faces
        
        e_dict = {}
        for e in edges:
            e0, e1 = min(e), max(e)
            e_dict[(e0, e1)] = []
        
        for f in faces:
            s = vs[f[1]] - vs[f[0]]
            t = vs[f[2]] - vs[f[1]]
            u = vs[f[0]] - vs[f[2]]
            cos_0 = np.inner(s, -u) / (np.linalg.norm(s) * np.linalg.norm(u))
            cos_1 = np.inner(t, -s) / (np.linalg.norm(t) * np.linalg.norm(s)) 
            cos_2 = np.inner(u, -t) / (np.linalg.norm(u) * np.linalg.norm(t))
            cot_0 = cos_0 / (np.sqrt(1 - cos_0 ** 2) + 1e-12)
            cot_1 = cos_1 / (np.sqrt(1 - cos_1 ** 2) + 1e-12)
            cot_2 = cos_2 / (np.sqrt(1 - cos_2 ** 2) + 1e-12)
            key_0 = (min(f[1], f[2]), max(f[1], f[2]))
            key_1 = (min(f[2], f[0]), max(f[2], f[0]))
            key_2 = (min(f[0], f[1]), max(f[0], f[1]))
            e_dict[key_0].append(cot_0)
            e_dict[key_1].append(cot_1)
            e_dict[key_2].append(cot_2)
        
        for e in e_dict:
            e_dict[e] = -0.5 * (e_dict[e][0] + e_dict[e][1])

        C_ind = [[], []]
        C_val = []
        ident = [0] * len(vs)
        for e in e_dict:
            C_ind[0].append(e[0])
            C_ind[1].append(e[1])
            C_ind[0].append(e[1])
            C_ind[1].append(e[0])
            C_val.append(e_dict[e])
            C_val.append(e_dict[e])
            ident[e[0]] += -1.0 * e_dict[e]
            ident[e[1]] += -1.0 * e_dict[e]
        Am_ind = torch.LongTensor(C_ind)
        Am_val = -1.0 * torch.FloatTensor(C_val)
        self.Am = torch.sparse.FloatTensor(Am_ind, Am_val, torch.Size([len(vs), len(vs)]))

        for i in range(len(vs)):
            C_ind[0].append(i)
            C_ind[1].append(i)
        
        C_val = C_val + ident
        C_ind = torch.LongTensor(C_ind)
        C_val = torch.FloatTensor(C_val)
        # cotangent matrix
        self.Lm = torch.sparse.FloatTensor(C_ind, C_val, torch.Size([len(vs), len(vs)]))
        self.Dm = torch.diag(torch.tensor(ident)).float().to_sparse()
        self.Lm_sym = torch.sparse.mm(torch.pow(self.Dm, -0.5), torch.sparse.mm(self.Lm, torch.pow(self.Dm, -0.5).to_dense())).to_sparse()
        #self.L = torch.sparse.mm(self.D_minus_half, torch.sparse.mm(C, self.D_minus_half.to_dense()))
        self.Am_I = (torch.eye(len(vs)) + self.Am).to_sparse()
        Dm_I_diag = torch.sum(self.Am_I.to_dense(), dim=1)
        self.Dm_I = torch.diag(Dm_I_diag).to_sparse()
        self.meshconvF = torch.sparse.mm(torch.pow(self.Dm_I, -0.5), torch.sparse.mm(self.Am_I, torch.pow(self.Dm_I, -0.5).to_dense())).to_sparse()

    def get_chebconv_coef(self, k=2):
        coef_list = []
        eig_max = torch.lobpcg(self.Lm_sym, k=1)[0][0]
        Lm_hat = -1.0 * torch.eye(len(self.vs)) + 2.0 * self.Lm_sym / eig_max.item()
        self.Lm_hat = Lm_hat.to_sparse()
        for i in range(k):
            if i == 0:
                coef = torch.eye(len(self.vs)).to_sparse()
                coef_list.append(coef)
            elif i == 1:
                coef_list.append(self.Lm_hat)
            else:
                coef = 2.0 * torch.sparse.mm(self.Lm_hat, coef_list[-1].to_dense()) - coef_list[-2]
                coef_list.append(coef.to_sparse())
        return coef_list
    
    def eigen_decomposition(self, L, k=100):
        L = L.to_dense().numpy()
        csr = sp.sparse.csr_matrix(L)
        w, v = sp.sparse.linalg.eigs(csr, which="SR", k=k)
        #index = np.argsort(np.real(w))[::-1]    # LR
        index = np.argsort(np.real(w))    # SR
        # vs_code can include either vertex code or face code
        vs_code = np.real(v[:, index]).astype(np.float)
        vs_code /= np.linalg.norm(vs_code, axis=0, keepdims=True)
        # import open3d as o3d
        # mesh = o3d.io.read_triangle_mesh(self.path)
        # for i in range(0, k, 1):
        #     colors = np.zeros([len(self.vs), 3])
        #     r = vs_code[:, i]
        #     r = (r+1)/2
        #     """
        #     r_min, r_max = np.min(r), np.max(r)
        #     r = (r - r_min) / (r_max-r_min+1.0e-16)
        #     """
        #     colors[:, 1] = r
        #     mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        #     o3d.io.write_triangle_mesh("eigen_decom/eigen_{}.obj".format(str(i)), mesh)
        # import pdb;pdb.set_trace()

        return vs_code
    
    def simplification(self, target_v, valence_aware=True):
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges

        """ 1. compute Q for each vertex """
        Q_s = [[] for _ in range(len(vs))]
        E_s = [[] for _ in range(len(vs))]
        for i, v in enumerate(vs):
            f_s = np.array(list(vf[i]), dtype=np.int)
            fc_s = fc[f_s]
            fn_s = fn[f_s]
            d_s = - 1.0 * np.sum(fn_s * fc_s, axis=1, keepdims=True)
            abcd_s = np.concatenate([fn_s, d_s], axis=1)
            Q_s[i] = np.matmul(abcd_s.T, abcd_s)

            v4 = np.concatenate([v, np.array([1])])
            E_s[i] = np.matmul(v4, np.matmul(Q_s[i], v4.T))

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            v_new = 0.5 * (v_0 + v_1)
            v4_new = np.concatenate([v_new, np.array([1])])
            valence_penalty = 1
            if valence_aware:
                merged_faces = vf[e[0]].intersection(vf[e[1]])
                valence_new = len(vf[e[0]].union(vf[e[1]]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)

            Q_0, Q_1 = Q_s[e[0]], Q_s[e[1]]
            Q_new = Q_0 + Q_1
            E_new = np.matmul(v4_new, np.matmul(Q_new, v4_new.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (e[0], e[1])))
        
        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]

        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                #print("non-manifold can be occured!!" , len(shared_vv))
                self.remove_tri_valance(simp_mesh, vi_0, vi_1, shared_vv, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap)
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                #print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap, valence_aware=valence_aware)
                #print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        simp_mesh.org = self
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh
    def edge_based_simplification(self, target_v, valence_aware=True):
        vs, vf, fn, fc, edges = self.vs, self.vf, self.fn, self.fc, self.edges
        edge_len = vs[edges][:,0,:] - vs[edges][:,1,:]
        edge_len = np.linalg.norm(edge_len, axis=1)
        edge_len_heap = np.stack([edge_len, np.arange(len(edge_len))], axis=1).tolist()
        heapq.heapify(edge_len_heap)

        """ 2. compute E for every possible pairs and create heapq """
        E_heap = []
        for i, e in enumerate(edges):
            v_0, v_1 = vs[e[0]], vs[e[1]]
            heapq.heappush(E_heap, (edge_len[i], (e[0], e[1])))
        
        """ 3. collapse minimum-error vertex """
        simp_mesh = copy.deepcopy(self)

        vi_mask = np.ones([len(simp_mesh.vs)]).astype(np.bool)
        fi_mask = np.ones([len(simp_mesh.faces)]).astype(np.bool)

        vert_map = [{i} for i in range(len(simp_mesh.vs))]

        while np.sum(vi_mask) > target_v:
            if len(E_heap) == 0:
                print("[Warning]: edge cannot be collapsed anymore!")
                break

            E_0, (vi_0, vi_1) = heapq.heappop(E_heap)

            if (vi_mask[vi_0] == False) or (vi_mask[vi_1] == False):
                continue

            """ edge collapse """
            shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
            merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vi_1])

            if len(shared_vv) != 2:
                """ non-manifold! """
                # print("non-manifold can be occured!!" , len(shared_vv))
                continue

            elif len(merged_faces) != 2:
                """ boundary """
                # print("boundary edge cannot be collapsed!")
                continue

            else:
                self.edge_based_collapse(simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap, valence_aware=valence_aware)
                # print(np.sum(vi_mask), np.sum(fi_mask))
        
        self.rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map)
        simp_mesh.simp = True
        self.build_hash(simp_mesh, vi_mask, vert_map)
        
        return simp_mesh

    @staticmethod
    def remove_tri_valance(simp_mesh, vi_0, vi_1, shared_vv, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap):
        return
        
    def edge_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, Q_s, E_heap, valence_aware):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int)] = False

        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])

        """ recompute E """
        Q_0 = Q_s[vi_0]
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])
            Q_1 = Q_s[vv_i]
            Q_new = Q_0 + Q_1
            v4_mid = np.concatenate([v_mid, np.array([1])])
            
            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(simp_mesh.vf[vi_0].union(simp_mesh.vf[vv_i]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)

            E_new = np.matmul(v4_mid, np.matmul(Q_new, v4_mid.T)) * valence_penalty
            heapq.heappush(E_heap, (E_new, (vi_0, vv_i)))
    
    def edge_based_collapse(self, simp_mesh, vi_0, vi_1, merged_faces, vi_mask, fi_mask, vert_map, E_heap, valence_aware):
        shared_vv = list(set(simp_mesh.v2v[vi_0]).intersection(set(simp_mesh.v2v[vi_1])))
        new_vi_0 = set(simp_mesh.v2v[vi_0]).union(set(simp_mesh.v2v[vi_1])).difference({vi_0, vi_1})
        simp_mesh.vf[vi_0] = simp_mesh.vf[vi_0].union(simp_mesh.vf[vi_1]).difference(merged_faces)
        simp_mesh.vf[vi_1] = set()
        simp_mesh.vf[shared_vv[0]] = simp_mesh.vf[shared_vv[0]].difference(merged_faces)
        simp_mesh.vf[shared_vv[1]] = simp_mesh.vf[shared_vv[1]].difference(merged_faces)

        simp_mesh.v2v[vi_0] = list(new_vi_0)
        for v in simp_mesh.v2v[vi_1]:
            if v != vi_0:
                simp_mesh.v2v[v] = list(set(simp_mesh.v2v[v]).difference({vi_1}).union({vi_0}))
        simp_mesh.v2v[vi_1] = []
        vi_mask[vi_1] = False

        vert_map[vi_0] = vert_map[vi_0].union(vert_map[vi_1])
        vert_map[vi_0] = vert_map[vi_0].union({vi_1})
        vert_map[vi_1] = set()
        
        fi_mask[np.array(list(merged_faces)).astype(np.int)] = False

        simp_mesh.vs[vi_0] = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vi_1])

        """ recompute E """
        for vv_i in simp_mesh.v2v[vi_0]:
            v_mid = 0.5 * (simp_mesh.vs[vi_0] + simp_mesh.vs[vv_i])
            edge_len = np.linalg.norm(simp_mesh.vs[vi_0] - simp_mesh.vs[vv_i])
            valence_penalty = 1
            if valence_aware:
                merged_faces = simp_mesh.vf[vi_0].intersection(simp_mesh.vf[vv_i])
                valence_new = len(simp_mesh.vf[vi_0].union(simp_mesh.vf[vv_i]).difference(merged_faces))
                valence_penalty = self.valence_weight(valence_new)
                edge_len *= valence_penalty

            heapq.heappush(E_heap, (edge_len, (vi_0, vv_i)))

    @staticmethod
    def valence_weight(valence_new):
        valence_penalty = abs(valence_new - OPTIM_VALENCE) * VALENCE_WEIGHT + 1
        if valence_new == 3:
            valence_penalty *= 100000
        return valence_penalty      
    
    @staticmethod
    def rebuild_mesh(simp_mesh, vi_mask, fi_mask, vert_map):
        face_map = dict(zip(np.arange(len(vi_mask)), np.cumsum(vi_mask)-1))
        simp_mesh.vs = simp_mesh.vs[vi_mask]
        
        vert_dict = {}
        for i, vm in enumerate(vert_map):
            for j in vm:
                vert_dict[j] = i

        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                if f[j] in vert_dict:
                    simp_mesh.faces[i][j] = vert_dict[f[j]]

        simp_mesh.faces = simp_mesh.faces[fi_mask]
        for i, f in enumerate(simp_mesh.faces):
            for j in range(3):
                simp_mesh.faces[i][j] = face_map[f[j]]
        
        simp_mesh.compute_face_normals()
        simp_mesh.compute_face_center()
        simp_mesh.build_gemm()
        simp_mesh.compute_vert_normals()
        simp_mesh.build_v2v()
        simp_mesh.build_vf()

    @staticmethod
    def build_hash(simp_mesh, vi_mask, vert_map):
        pool_hash = {}
        unpool_hash = {}
        for simp_i, idx in enumerate(np.where(vi_mask)[0]):
            if len(vert_map[idx]) == 0:
                print("[ERROR] parent node cannot be found!")
                return
            for org_i in vert_map[idx]:
                pool_hash[org_i] = simp_i
            unpool_hash[simp_i] = list(vert_map[idx])
        
        """ check """
        vl_sum = 0
        for vl in unpool_hash.values():
            vl_sum += len(vl)

        if (len(set(pool_hash.keys())) != len(vi_mask)) or (vl_sum != len(vi_mask)):
            print("[ERROR] Original vetices cannot be covered!")
            return
        
        pool_hash = sorted(pool_hash.items(), key=lambda x:x[0])
        simp_mesh.pool_hash = pool_hash
        simp_mesh.unpool_hash = unpool_hash
    
    @staticmethod
    def mesh_merge(lap, org_mesh, new_pos, preserve, w=10, w_b=0):
        org_pos = torch.from_numpy(org_mesh.vs).float()
        preserve_boundary = torch.sparse.mm(org_mesh.AdjI.float(), 1-preserve.float().reshape(-1, 1)) == 0
        # preserve_boundary = torch.sparse.mm(org_mesh.AdjI.float(), (1-preserve_boundary.float())) == 0
        preserve_boundary = preserve_boundary.reshape(-1)
        boundary = torch.logical_xor(preserve, preserve_boundary)
        A_cat = torch.eye(len(org_pos))[preserve_boundary] * w
        A_boundary = torch.eye(len(org_pos))[boundary] * w_b
        A = torch.cat([lap.to_dense(), A_cat], dim=0)
        A = torch.cat([A, A_boundary], dim=0).to_sparse()
        b_org = torch.sparse.mm(lap, new_pos)
        b_cat = org_pos[preserve_boundary] * w
        b_boundary = org_pos[boundary] * w_b
        b = torch.cat([b_org, b_cat], dim=0)
        b = torch.cat([b, b_boundary], dim=0)
        AtA = torch.sparse.mm(A.t(), A.to_dense())
        Atb = torch.sparse.mm(A.t(), b)
        ref_pos, _ = torch.solve(Atb, AtA)
        # AtA_inv = torch.inverse(AtA)
        # new_pos = torch.matmul(AtA_inv, torch.sparse.mm(A.T, b))
        return ref_pos

    @staticmethod
    def copy_attribute(src_mesh, dst_mesh):
        dst_mesh.vf, dst_mesh.edges, dst_mesh.ve = src_mesh.vf, src_mesh.edges, src_mesh.ve
        dst_mesh.v2v = src_mesh.v2v
        
    def save(self, filename, color=False):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()
        v_colors = np.array(self.vc, dtype=np.float32).flatten()

        with open(filename, 'w') as fp:
            # Write positions
            if len(v_colors) == 0 or color == False:
                for i in range(0, vertices.size, 3):
                    x = vertices[i + 0]
                    y = vertices[i + 1]
                    z = vertices[i + 2]
                    fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))
            
            else:
                for i in range(0, vertices.size, 3):
                    x = vertices[i + 0]
                    y = vertices[i + 1]
                    z = vertices[i + 2]
                    c1 = v_colors[i + 0]
                    c2 = v_colors[i + 1]
                    c3 = v_colors[i + 2]
                    fp.write('v {0:.8f} {1:.8f} {2:.8f} {3:.8f} {4:.8f} {5:.8f}\n'.format(x, y, z, c1, c2, c3))

            # Write indices
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))
    
    def save_as_ply(self, filename, fn):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()
        fnormals = np.array(fn, dtype=np.float32).flatten()

        with open(filename, 'w') as fp:
            # Write Header
            fp.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(self.vs)))
            fp.write("property float x\nproperty float y\nproperty float z\n")
            fp.write("element face {}\n".format(len(self.faces)))
            fp.write("property list uchar int vertex_indices\n")
            fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
            fp.write("end_header\n")
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write("{0:.6f} {1:.6f} {2:.6f}\n".format(x, y, z))
            
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0]
                i1 = indices[i + 1]
                i2 = indices[i + 2]
                c0 = fnormals[i + 0]
                c1 = fnormals[i + 1]
                c2 = fnormals[i + 2]
                c0 = np.clip(int(255 * c0), 0, 255)
                c1 = np.clip(int(255 * c1), 0, 255)
                c2 = np.clip(int(255 * c2), 0, 255)
                c3 = 255
                fp.write("3 {0} {1} {2} {3} {4} {5} {6}\n".format(i0, i1, i2, c0, c1, c2, c3))

    def display_face_normals(self, fn):
        import open3d as o3d
        self.compute_face_center()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.fc))
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(fn))
        o3d.visualization.draw_geometries([pcd])
                