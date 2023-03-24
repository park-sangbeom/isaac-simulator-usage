import os
import numpy as np
import mujoco

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def rpy2r(rpy):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy[0]
    pitch = rpy[1]
    yaw   = rpy[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

class MJCFParserClass(object):
    """
        MJCF Parser class
    """

    def __init__(self, name='Robot', rel_xml_path=None, VERBOSE=True):
        """
            Initialize MJCF parser
        """
        self.name = name
        self.rel_xml_path = rel_xml_path
        self.VERBOSE = VERBOSE
        # Constants
        self.tick = 0
        self.render_tick = 0
        # Parse an xml file
        if self.rel_xml_path is not None:
            self._parse_xml()
        # Reset
        self.reset()
        # Print
        if self.VERBOSE:
            self.print_info()

    def _parse_xml(self):
        """
            Parse an xml file
        """
        self.full_xml_path = os.path.abspath(
            os.path.join(os.getcwd(), self.rel_xml_path))
        self.model = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_geom = self.model.ngeom  # number of geometries
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, x)
                           for x in range(self.model.ngeom)]
        self.n_body = self.model.nbody  # number of bodies
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, x)
                           for x in range(self.n_body)]
        self.n_dof = self.model.nv  # degree of freedom
        self.n_joint = self.model.njnt     # number of joints
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtJoint.mjJNT_HINGE, x)
                            for x in range(self.n_joint)]
        self.joint_types = self.model.jnt_type  # joint types
        self.joint_ranges = self.model.jnt_range  # joint ranges
        self.rev_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_HINGE)[
            0].astype(np.int32)
        self.rev_joint_names = [self.joint_names[x]
                                for x in self.rev_joint_idxs]
        self.n_rev_joint = len(self.rev_joint_idxs)
        self.rev_joint_mins = self.joint_ranges[self.rev_joint_idxs, 0]
        self.rev_joint_maxs = self.joint_ranges[self.rev_joint_idxs, 1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        self.pri_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_SLIDE)[
            0].astype(np.int32)
        self.pri_joint_names = [self.joint_names[x]
                                for x in self.pri_joint_idxs]
        self.pri_joint_mins = self.joint_ranges[self.pri_joint_idxs, 0]
        self.pri_joint_maxs = self.joint_ranges[self.pri_joint_idxs, 1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.n_pri_joint = len(self.pri_joint_idxs)
        # Actuator
        self.n_ctrl = self.model.nu  # number of actuators (or controls)
        self.ctrl_names = []
        for addr in self.model.name_actuatoradr:
            ctrl_name = self.model.names[addr:].decode().split('\x00')[0]
            self.ctrl_names.append(ctrl_name)  # get ctrl name
        self.ctrl_joint_idxs = []
        for ctrl_idx in range(self.n_ctrl):
            transmission_idx = self.model.actuator(
                self.ctrl_names[ctrl_idx]).trnid  # transmission index
            # index of the joint when the actuator acts on a joint
            joint_idx = self.model.jnt_qposadr[transmission_idx][0]
            self.ctrl_joint_idxs.append(joint_idx)
        self.ctrl_ranges = self.model.actuator_ctrlrange  # control range

    def print_info(self):
        """
            Printout model information
        """
        print("n_body:[%d]" % (self.n_geom))
        print("geom_names:%s" % (self.geom_names))
        print("n_body:[%d]" % (self.n_body))
        print("body_names:%s" % (self.body_names))
        print("n_joint:[%d]" % (self.n_joint))
        print("joint_names:%s" % (self.joint_names))
        print("joint_types:%s" % (self.joint_types))
        print("joint_ranges:\n%s" % (self.joint_ranges))
        print("n_rev_joint:[%d]" % (self.n_rev_joint))
        print("rev_joint_idxs:%s" % (self.rev_joint_idxs))
        print("rev_joint_names:%s" % (self.rev_joint_names))
        print("rev_joint_mins:%s" % (self.rev_joint_mins))
        print("rev_joint_maxs:%s" % (self.rev_joint_maxs))
        print("rev_joint_ranges:%s" % (self.rev_joint_ranges))
        print("n_pri_joint:[%d]" % (self.n_pri_joint))
        print("pri_joint_idxs:%s" % (self.pri_joint_idxs))
        print("pri_joint_names:%s" % (self.pri_joint_names))
        print("pri_joint_mins:%s" % (self.pri_joint_mins))
        print("pri_joint_maxs:%s" % (self.pri_joint_maxs))
        print("pri_joint_ranges:%s" % (self.pri_joint_ranges))
        print("n_ctrl:[%d]" % (self.n_ctrl))
        print("ctrl_names:%s" % (self.ctrl_names))
        print("ctrl_joint_idxs:%s" % (self.ctrl_joint_idxs))
        print("ctrl_ranges:\n%s" % (self.ctrl_ranges))

    def reset(self):
        """
            Reset
        """
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.render_tick = 0

    def step(self, ctrl=None, ctrl_idxs=None, nstep=1):
        """
            Forward dynamics
        """
        if ctrl is not None:
            if ctrl_idxs is None:
                self.data.ctrl[:] = ctrl
            else:
                self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=nstep)
        self.tick = self.tick + 1

    def forward(self, q=None, joint_idxs=None):
        """
            Forward kinematics
        """
        if q is not None:
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q
            else:
                self.data.qpos = q
        mujoco.mj_forward(self.model, self.data)
        self.tick = self.tick + 1

    def render(self, render_every=1):
        """
            Render
        """
        if self.USE_MUJOCO_VIEWER:
            if ((self.render_tick % render_every) == 0) or (self.render_tick == 0):
                self.viewer.render()
            self.render_tick = self.render_tick + 1
        else:
            print("[%s] Viewer NOT initialized." % (self.name))

    def get_p_body(self, body_name):
        """
            Get body position
        """
        return self.data.body(body_name).xpos

    def get_R_body(self, body_name):
        """
            Get body rotation matrix
        """
        return self.data.body(body_name).xmat.reshape([3, 3])

    def get_pR_body(self, body_name):
        """
            Get body position and rotation matrix
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p, R

    def get_q(self, joint_idxs=None):
        """
            Get joint position in (radian)
        """
        if joint_idxs is None:
            q = self.data.qpos
        else:
            q = self.data.qpos[joint_idxs]
        return q

    def get_J_body(self, body_name):
        """
            Get Jocobian matrices of a body
        """
        J_p = np.zeros((3, self.model.nv))  # nv: nDoF
        J_R = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_p, J_R,
                          self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p, J_R]))
        return J_p, J_R, J_full

    def get_ik_ingredients(self, body_name, p_trgt=None, R_trgt=None, IK_P=True, IK_R=True):
        """
            Get IK ingredients
        """
        J_p, J_R, J_full = self.get_J_body(body_name=body_name)
        p_curr, R_curr = self.get_pR_body(body_name=body_name)
        print("R_curr",R_curr.shape)
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr, R_trgt)
            w_err = R_curr @ r2w(R_err)
            J = J_full
            err = np.concatenate((p_err, w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J = J_p
            err = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr, R_trgt)
            w_err = R_curr @ r2w(R_err)
            J = J_R
            err = w_err
        else:
            J = None
            err = None
        return J, err

    def damped_ls(self, J, err, eps=1e-6, stepsize=1.0, th=5*np.pi/180.0):
        """
            Dampled least square for IK
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J) +
                                      eps*np.eye(J.shape[1]), b=J.T@err)
        dq = trim_scale(x=dq, th=th)
        return dq

    def onestep_ik(self, body_name, p_trgt=None, R_trgt=None, IK_P=True, IK_R=True,
                   joint_idxs=None, stepsize=1, eps=1e-1, th=5*np.pi/180.0):
        """
            Solve IK for a single step
        """
        J, err = self.get_ik_ingredients(
            body_name=body_name, p_trgt=p_trgt, R_trgt=R_trgt, IK_P=IK_P, IK_R=IK_R)
        dq = self.damped_ls(J, err, stepsize=stepsize, eps=eps, th=th)
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        q = self.get_q(joint_idxs=joint_idxs)
        q = q + dq[joint_idxs]
        # FK
        self.forward(q=q, joint_idxs=joint_idxs)
        return q, err

    def get_body_names(self, prefix='obj_'):
        """
            Get body names with prefix
        """
        body_names = [x for x in self.body_names if x[:len(prefix)] == prefix]
        return body_names

