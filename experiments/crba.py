import pinocchio as pino
import numpy as np
from system_identification.crba_rotor_py import crba_rotor
"""
    This file manually calculate composite rigid body algorithms for joint space inertia,
    the algorithm ref to chapter 6.2 of the book Rigid body dynamics algorithm by Roy Featherstone.
    
    Note that in pinocchio:
        1) Transformation matrix is defined from child link to parent link, liMi[i].rotation, 
           and liMi[i].translation are defined from link i to link i-1;
        2) The spatial motion and force vector are defined differently compare with the definition
           in the book. In the book, [rotation, translation]^{T}, but in pinocchio, [translation,
           rotation]^{T}
"""

def skewSym(x):
    return np.array([
                        [0., -x[2], x[1]],
                        [x[2], 0., -x[0]],
                        [-x[1], x[0], 0.]
                    ])

def spatialInertia(Ic, m, c):
    # Ic: 3X3 tensor, inertia on the CoM frame: Io = Ic + mcX cX^{T}, X denotes cross product
    # m: scalar, mass
    # c: 1X3 tensor, CoM
    # Io: 6X6 tensor
    Io = np.block([
                        [Ic+m*np.dot(skewSym(c), skewSym(c).T), m*skewSym(c)],
                        [m*skewSym(c).T, m*np.identity(len(Ic))]
                  ])
    return Io

def spatialInertiaBody(Ib, m, c):
    # Ic: 3X3 tensor, inertia on the CoM frame: Io = Ic + mcX cX^{T}, X denotes cross product
    # m: scalar, mass
    # c: 1X3 tensor, CoM
    # Io: 6X6 tensor
    Io = np.block([
                        [Ib, m*skewSym(c)],
                        [m*skewSym(c).T, m*np.identity(len(Ib))]
                  ])
    return Io

def spatialInertiaBodyPino(Ib_vec, m, c):
    # Ic: 3X3 tensor, inertia on the CoM frame: Io = Ic + mcX cX^{T}, X denotes cross product
    # m: scalar, mass
    # c: 1X3 tensor, CoM
    # Io: 6X6 tensor
    Ixx, Ixy, Iyy, Ixz, Iyz, Izz = Ib_vec
    Ib = np.array([
                    [Ixx, Ixy, Ixz],
                    [Ixy, Iyy, Iyz],
                    [Ixz, Iyz, Izz]
    ])
    Io = np.block([
                        [m*np.identity(len(c)), m*skewSym(c).T],
                        [m*skewSym(c), Ib]
                  ])
    return Io

def spatialMotionTransform(E, r):
    # E: the rotation matrix from child link to parent link, express in the child frame;
    # r: the translation vector from child link to parent link, express in the child frame;
    # Note that the implementation assumes spatial velocity as [translation, rotation]^{T}
    # instead of the classic way of [rotation, translation]^{T}
    # XMotion: transformation matrix for motion from parent to child
    SpatialRotation = np.block([
                                    [E.T, np.zeros_like(E)],
                                    [np.zeros_like(E), E.T]
                                ])
    SpatialTranslation = np.block([
                                    [np.identity(len(E)), -skewSym(r)],
                                    [np.zeros_like(E), np.identity(len(E))]
                                  ])
    XMotion = np.dot(SpatialRotation, SpatialTranslation)
    return XMotion

def spatialMotionTransformInverse(E, r):
    # E: the rotation matrix from child link to parent link, express in the child frame
    # r: the translation vector from child link to parent link, express in the child frame;
    # XMotionInv: transformation matrix for motion from child to parent
    SpatialRotation = np.block([
                                    [E, np.zeros_like(E)],
                                    [np.zeros_like(E), E]
                                ])
    SpatialTranslation = np.block([
                                    [np.identity(len(E)), skewSym(r)],
                                    [np.zeros_like(E), np.identity(len(E))]
                                  ])
    XMotionInv = np.dot(SpatialTranslation, SpatialRotation)
    return XMotionInv

def spatialForceTransform(E, r):
    # E: the rotation matrix from child link to parent link, express in the child frame;
    # r: the translation vector from child link to parent link, express in the child frame;
    # Note that the implementation assumes spatial velocity as [force, torque]^{T}
    # instead of the classic way of [force, torque]^{T}
    # XForce: transformation matrix for force from parent to child
    SpatialRotation = np.block([
                                    [E.T, np.zeros_like(E)],
                                    [np.zeros_like(E), E.T]
                                ])
    SpatialTranslation = np.block([
                                    [np.identity(len(E)), np.zeros_like(E)],
                                    [-skewSym(r), np.identity(len(E))]
                                  ])
    XForce = np.dot(SpatialRotation, SpatialTranslation)
    return XForce

def spatialForceTransformInverse(E, r):
    # E: the rotation matrix from child link to parent link, express in the child frame;
    # r: the translation vector from child link to parent link, express in the child frame;
    # XForcecInv: transformation matrix for force from child to parent
    SpatialRotation = np.block([
                                    [E, np.zeros_like(E)],
                                    [np.zeros_like(E), E]
                                ])
    SpatialTranslation = np.block([
                                    [np.identity(len(E)), np.zeros_like(E)],
                                    [skewSym(r), np.identity(len(E))]
                                  ])
    XForceInv = np.dot(SpatialTranslation, SpatialRotation)
    return XForceInv

class PinoManger(object):
    def __init__(self, urdf_file):
        self.pino_model = pino.buildModelFromUrdf(urdf_file)
        self.pino_data = self.pino_model.createData()
        self.inertias = self.pino_model.inertias.tolist()
        self._pre_cal()

    def _pre_cal(self):
        q = np.zeros(7)
        pino.forwardKinematics(self.pino_model, self.pino_data, q)
        self.Si = []
        for jointi in range(len(q)):
            Si = pino.computeJointJacobian(self.pino_model, self.pino_data, q, jointi+1)[:, jointi] # skip the base link
            self.Si.append(Si)

    def _getSpatialRotAxis(self, joint_id):
        return self.Si[joint_id]

    def _getSpatialLinkInertia(self, joint_id):
        # Note that pinocchio use a different order compare with the one in the book
        # In pinocchio,
        #     Io = [
        #              [m*identity(3), mcX^{T}],
        #              [mcX, Io]
        #          ]
        # In the book of Roy Featherstone,
        #   Io = [
        #           [Io, mcX],
        #           [mcX^{T}, Io]
        #        ]
        pinoInertiaObj = self.inertias[joint_id + 1] # skip the universal joint
        return pinoInertiaObj.matrix()

    def _getForceTransform(self, joint_id, inverse=False):
        # calculate 6X6 transformation matrix for motion, by default parent to child
        # frame, e.g. from joint_id-1 to joint_id
        pinoRotTrans = self.pino_data.liMi.tolist()[joint_id + 1] # skip the universal joint
        E, r = pinoRotTrans.rotation, pinoRotTrans.translation
        if not inverse:
            forceTransform = spatialForceTransform(E, r)
        else:
            forceTransform = spatialForceTransformInverse(E, r)
        return forceTransform

    def _getMotionTransform(self, joint_id, inverse=False):
        # obtain 6X6 transformation matrix for motion, in pinocchio, the motion transformation
        # is by default from child to parent frame, e.g. from joint_id to joint_id-1, actionInverse
        # from parent to child, and action from child to parent. Here we change to by default parent
        # to child frame, e.g. from joint_id-1 to joint_id
        if not inverse:
            motionTransform = self.pino_data.liMi[joint_id + 1].actionInverse
        else:
            motionTransform = self.pino_data.liMi[joint_id + 1].action
        return motionTransform

    def cal_mass_entry(self, jointi, jointj, q):
        # check whether input is valid or not
        assert jointi <= jointj and len(q) == self.pino_model.nq, \
            f"jointi should be smaller than joint j, q should with the shape of {self.pino_model.nq}"

        # calculate forward kinematics to update data
        pino.forwardKinematics(self.pino_model, self.pino_data, q)

        # calculate composite rigid body inertia
        Ij_composite = np.zeros(shape=(6, 6))
        for i in range(6, jointj - 1, -1):
            if i < 6:
                # here add 1 because: the transformation starts from child to parent
                M_i_mui = self._getMotionTransform(i + 1, inverse=False)
                F_mui_i = self._getForceTransform(i + 1, inverse=True)
            else:
                # transformation for joint7, mui denotes the child of joint i
                M_i_mui = np.identity(6)
                F_mui_i = np.identity(6)

            Ii = self._getSpatialLinkInertia(i)
            Ij_composite = Ii + np.linalg.multi_dot([F_mui_i, Ij_composite, M_i_mui])

        # calculate transformation matrix for force tensor from frame j to i
        F_ji = np.identity(6)
        for id in range(jointj, jointi, -1):
            E, r = self.pino_data.liMi[id + 1].rotation, self.pino_data.liMi[id + 1].translation
            F_id = spatialForceTransformInverse(E, r)
            F_ji = np.dot(F_id, F_ji)

        # retrieve spatial rotating axis from pinocchio
        Si = self.Si[jointi]
        Sj = self.Si[jointj]
        M_ij = np.linalg.multi_dot([Si.T, F_ji, Ij_composite, Sj])
        return M_ij

    def gen_rotor_inertia(self, rotor_param):
        """
        Generate 6X6 spatial inertia for CRBA
        I = [
                [m*identity(3), mcX^{T}],
                [mcX, Io]
             ]
        There are two possible parameterization for the rotor
        1) overall 21 parameters for the rotor of each joint,
           you have 3 parameters, Ixz, Iyz, Izz;
        2) overall 7 parameters for the rotor of each joint,
           you have 1 parameter Izz,
        Here we only implement the Izz case
        """
        self.I_spatial_rotor = []
        for i in range(len(rotor_param)):
            Izz = rotor_param[i]

            # 3X3 inertia
            I_rotor_o = np.array([
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., Izz]
                                 ])
            I_o_ = np.block([
                                    [np.zeros(shape=(3, 3)), np.zeros(shape=(3, 3))],
                                    [np.zeros(shape=(3, 3)), I_rotor_o]
                             ])
            self.I_spatial_rotor.append(I_o_)

    def _getSpatialRotorInertia(self, joint_id):
        # Note that pinocchio use a different order compare with the one in the book
        # In pinocchio,
        #     Io = [
        #              [zero(3, 3), zero(3, 3)],
        #              [zero(3, 3), I_rotor_o]
        #          ]
        return self.I_spatial_rotor[joint_id]

    def _gen_gear_ratio_matrix(self, gear_ratio):
        # Generate matrix for gear ratio integration
        rho1, rho2, rho3, rho4, rho5, rho6, rho7 = gear_ratio
        upper_tri = np.array([
                                [0.5*rho1**2, -rho2, -rho3, -rho4, -rho5, -rho6, -rho7],
                                [0., 0.5*rho2**2, -rho3, -rho4, -rho5, -rho6, -rho7],
                                [0., 0., 0.5*rho3**2, -rho4, -rho5, -rho6, -rho7],
                                [0., 0., 0., 0.5*rho4**2, -rho5, -rho6, -rho7],
                                [0., 0., 0., 0., 0.5*rho5**2, -rho6, -rho7],
                                [0., 0., 0., 0., 0., 0.5*rho6**2, -rho7],
                                [0., 0., 0., 0., 0., 0., 0.5*rho7**2]
        ])
        rho_mat = upper_tri + upper_tri.T
        return rho_mat

    def crba_joint_fromDynParam(self, q, dyn_param):
        """
        In the function of cal_mass_entry, we assume: jointi < jointj,
        In the book of Roy Featherstone, the algorithm is based on
        jointi > jointj. Here we again assume: jointi > jointj, aligned
        with the algorithm in the book, can also be implemented the
        other way around
        """
        H = np.zeros(shape=(len(q), len(q)))
        nBody = len(q)
        Ics = []

        # first we initialize all composite inertia to body inertia
        # generate spatial link inertia
        for jointid in range(nBody):
            dyn_param_i = dyn_param[jointid*10: (jointid+1)*10]
            m, h, Ib_vec = dyn_param_i[0], dyn_param_i[1: 4], dyn_param_i[4:]
            c = h / m
            Ics.append(spatialInertiaBodyPino(Ib_vec, m, c))

        # outer loop to update the composite inertia for the upper link i
        # while inner loop calculate the composite force and the entry of
        # joint-space mass matrix
        for i in range(nBody-1, -1, -1):
            # Update composite
            if i < 6:
                M_i_mui = self._getMotionTransform(i + 1, inverse=False)
                F_mui_i = self._getForceTransform(i + 1, inverse=True)
                Ic_iplus1 = Ics[i+1]
                Ics[i] = Ics[i] + F_mui_i @ Ic_iplus1 @ M_i_mui
            else:
                # joint 7 doesn't have child
                pass
            Si = self._getSpatialRotAxis(i)
            for j in range(i, -1, -1):
                if i == j:
                    F = np.dot(Ics[i], Si)
                    Hii = np.dot(Si, F)
                    H[i][i] = Hii
                else:
                    Sj = self._getSpatialRotAxis(j)
                    F_mui_i = self._getForceTransform(j + 1, inverse=True)
                    F = np.dot(F_mui_i, F)
                    Hij = np.dot(F.T, Sj)
                    H[i][j] = Hij
                    H[j][i] = Hij
        return H

    def crba_joint(self, q):
        """
        In the function of cal_mass_entry, we assume: jointi < jointj,
        In the book of Roy Featherstone, the algorithm is based on
        jointi > jointj. Here we again assume: jointi > jointj, aligned
        with the algorithm in the book, can also be implemented the
        other way around
        """
        H = np.zeros(shape=(len(q), len(q)))
        nBody = len(q)
        Ics = []

        # first we initialize all composite inertia to body inertia
        for jointid in range(nBody):
            Ics.append(self._getSpatialLinkInertia(jointid))

        # outer loop for update the composite inertia for the upper link i
        # while inner loop calculate the composite force and the entry of
        # joint-space mass matrix
        for i in range(nBody-1, -1, -1):
            # Update composite
            if i < 6:
                M_i_mui = self._getMotionTransform(i + 1, inverse=False)
                F_mui_i = self._getForceTransform(i + 1, inverse=True)
                Ic_iplus1 = Ics[i+1]
                Ics[i] = Ics[i] + F_mui_i @ Ic_iplus1 @ M_i_mui
            else:
                # joint 7 doesn't have child
                pass
            Si = self._getSpatialRotAxis(i)
            for j in range(i, -1, -1):
                if i == j:
                    F = np.dot(Ics[i], Si)
                    Hii = np.dot(Si, F)
                    H[i][i] = Hii
                else:
                    Sj = self._getSpatialRotAxis(j)
                    F_mui_i = self._getForceTransform(j + 1, inverse=True)
                    F = np.dot(F_mui_i, F)
                    Hij = np.dot(F.T, Sj)
                    H[i][j] = Hij
                    H[j][i] = Hij
        return H

    def crba_rotor(self, q, rotor_param):
        """
        In the function of cal_mass_entry, we assume: jointi < jointj,
        In the book of Roy Featherstone, the algorithm is based on
        jointi > jointj. Here we again assume: jointi > jointj, aligned
        with the algorithm in the book, can also be implemented the
        other way around
        """
        # update the spatial rotor inertia with rotor parameters
        self.gen_rotor_inertia(rotor_param)

        H = np.zeros(shape=(len(q), len(q)))
        nBody = len(q)
        Ics = []

        # first we initialize all composite inertia to body inertia
        for jointid in range(nBody):
            Ics.append(self._getSpatialRotorInertia(jointid))

        # outer loop for update the composite inertia for the upper link i
        # while inner loop calculate the composite force and the entry of
        # joint-space mass matrix
        for i in range(nBody-1, -1, -1):
            # Update composite
            if i < 6:
                M_i_mui = self._getMotionTransform(i + 1, inverse=False)
                F_mui_i = self._getForceTransform(i + 1, inverse=True)
                Ic_iplus1 = Ics[i+1]
                Ics[i] = Ics[i] + F_mui_i @ Ic_iplus1 @ M_i_mui
            else:
                # joint 7 doesn't have child
                pass
            Si = self._getSpatialRotAxis(i)
            for j in range(i, -1, -1):
                if i == j:
                    F = np.dot(Ics[i], Si)
                    Hii = np.dot(Si, F)
                    H[i][i] = Hii
                else:
                    Sj = self._getSpatialRotAxis(j)
                    F_mui_i = self._getForceTransform(j + 1, inverse=True)
                    F = np.dot(F_mui_i, F)
                    Hij = np.dot(F.T, Sj)
                    H[i][j] = Hij
                    H[j][i] = Hij
        return H

    def crba_composite(self, q, rotor_param, gear_ratio):
        rho_mat = self._gen_gear_ratio_matrix(gear_ratio)
        mass_mat_joint = self.cal_mass_pino(q)
        mass_mat_rotor = self.crba_rotor(q, rotor_param)
        intermediate_result = rho_mat * mass_mat_rotor
        mass_mat_full = mass_mat_joint + rho_mat * mass_mat_rotor # element-wise product for two matrices
        return mass_mat_full

    def crba_composite_rotor_gear(self, q, rotor_param, gear_ratio):
        rho_mat = self._gen_gear_ratio_matrix(gear_ratio)
        mass_mat_rotor = self.crba_rotor(q, rotor_param)
        gearTimesRotor = rho_mat * mass_mat_rotor
        return gearTimesRotor

    def crba_composite_rotor_tau(self, q, ddq, rotor_param, gear_ratio):
        rho_mat = self._gen_gear_ratio_matrix(gear_ratio)
        mass_mat_rotor = self.crba_rotor(q, rotor_param)
        gearTimesRotor = rho_mat * mass_mat_rotor
        rotor_tau = gearTimesRotor @ ddq # element-wise product for two matrices
        return rotor_tau

    def cal_mass_pino(self, q):
        pino.crba(self.pino_model, self.pino_data, q)
        M = self.pino_data.M.copy()
        return M

def cal_M_ij(urdf_file, jointi, jointj):
    pino_model = pino.buildModelFromUrdf(urdf_file)
    pino_data = pino_model.createData()

    pino.forwardKinematics(pino_model, pino_data, q_0)
    Ij_composite = np.zeros(shape=(6, 6)) # start from joint 7
    for i in range(6, jointj-1, -1):
        if i < 6:
            # here add 2 because of: 1) 0 is the base link; 2) the transformation starts
            # from child to parent
            # M_mu_inverse = pino_data.liMi.tolist()[i + 2].actionInverse
            # actionInverse from parent to child, and action from child to parent
            E, r = pino_data.liMi[i + 2].rotation, pino_data.liMi[i + 2].translation
            M_mu_inverse = spatialMotionTransform(E, r)
            F_inv = spatialForceTransformInverse(E, r)
        else:
            M_mu_inverse = np.identity(6)
            F_inv = np.identity(6)
        Ii = pino_model.inertias.tolist()[i + 1].matrix()
        Ij_composite = Ii + np.linalg.multi_dot([F_inv, Ij_composite, M_mu_inverse])
    pino.crba(pino_model, pino_data, q_0)
    Ij_composite_pino = pino_data.Ycrb[jointj+1].matrix()
    F_ji = np.identity(6)
    for id in range(jointj, jointi, -1):
        E, r = pino_data.liMi[id + 1].rotation, pino_data.liMi[id + 1].translation
        F_id = spatialForceTransformInverse(E, r)
        F_ji = np.dot(F_id, F_ji)

    # simulate to check entry M_ij
    pino.forwardKinematics(pino_model, pino_data, q_0)
    Si = pino.computeJointJacobian(pino_model, pino_data, q_0, jointi)[:, jointi-1]
    Sj = pino.computeJointJacobian(pino_model, pino_data, q_0, jointj)[:, jointj-1]
    M_ij = np.linalg.multi_dot([Si.T, F_ji, Ij_composite, Sj])
    return M_ij

if __name__ == "__main__":
    URDF_FILE = f"../robot_description/iiwas_description/urdf/iiwas14_lmi.urdf"
    q_0 = np.zeros(7, dtype=np.float64)
    q_0[3] = np.random.uniform(-np.pi/2, np.pi/2)
    q_0[4] = np.random.uniform(-np.pi/2, np.pi/2)
    q_0[5] = np.random.uniform(-np.pi / 2, np.pi / 2)
    q_0[6] = np.random.uniform(-np.pi / 2, np.pi / 2)
    jointi = 1
    jointj = 5

    # manually compute M[i][j] and compare the result with pinocchio
    M_ij = cal_M_ij(URDF_FILE, jointi, jointj)
    pino_model = pino.buildModelFromUrdf(URDF_FILE)
    pino_data = pino_model.createData()
    pino.crba(pino_model, pino_data, q_0)
    M_ij_pino = pino_data.M.copy()[jointi][jointj]

    pinoManager = PinoManger(URDF_FILE)
    M_ij_class = pinoManager.cal_mass_entry(jointi, jointj, q_0)
    H = pinoManager.crba_joint(q_0)

    dyn_params = np.hstack([pino_model.inertias[i+1].toDynamicParameters() for i in range(7)])
    H_dyn = pinoManager.crba_joint_fromDynParam(q_0, dyn_params)
    M = pino_data.M.copy()
    pinoManagerCpp = crba_rotor()
    pinoManagerCpp.readURDF(URDF_FILE)
    H_dyn_cpp = pinoManagerCpp.crbaJointFromDynParam(q_0, dyn_params)

    rotor_param = np.ones(7) * 1e-5
    # rotor_param[3] = 1e-8
    # rotor_param[3] = 1e-2
    # rotor_param[5] = 1e-4
    gear_ratio = np.ones(7) * 100
    # gear_ratio[1] = 10
    # gear_ratio[3] = 100000

    # rotor_param = [0.001192, 0.003228, 0.000004929, 0.00001982, 0.00000471, 0.00001015, 0.000001173]
    # gear_ratio = [48.761, 30.802, 292.303, 158.319, 226.089, 115.666, 330.461]

    H_rotor = pinoManager.crba_rotor(q_0, rotor_param)
    H_rotor_gear = pinoManager.crba_composite_rotor_gear(q_0, rotor_param, gear_ratio)
    H_comp_joint_rotor = pinoManager.crba_composite(q_0, rotor_param, gear_ratio)

    pinoManagerCpp = crba_rotor()
    pinoManagerCpp.readURDF(URDF_FILE)
    H_comp_joint_rotor_cpp = pinoManagerCpp.crbaComposite(q_0, rotor_param, gear_ratio)

    q_0 = np.ones(shape=(10, 7), dtype=np.float64)
    ddq_0 = np.ones(shape=(10, 7), dtype=np.float64)
    list_rotor_taus = pinoManagerCpp.crbaRotorTorqueBatch(q_0, ddq_0, rotor_param, gear_ratio)
    rotor_tau_cpp = list_rotor_taus[0]
    pino.forwardKinematics(pinoManager.pino_model, pinoManager.pino_data, q_0[0])
    rotor_tau = pinoManager.crba_composite_rotor_tau(q_0[0], ddq_0[0], rotor_param, gear_ratio)
