import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import uniform


class TreeNode:
    def __init__(self, left = None, right = None, split = None, pleft = None, left_bound = None, right_bound = None, theta = None, depth = None):
        self.left = left
        self.right = right
        self.split = split
        self.pleft = pleft
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.theta = theta
        self.depth = depth
    def treecdf_A(self, x):
        if x <= self.left_bound or x > self.right_bound:
            return None
        mu_left = (self.split - self.left_bound) / (self.right_bound - self.left_bound)
        if not self.split or not self.pleft:
            return x
        if x <= self.split:
            return self.pleft/mu_left * (x - self.left_bound) + self.left_bound
        else:
            mu_right = 1 - mu_left
            return (1 - self.pleft)/mu_right * (x - self.right_bound) + self.right_bound
    def treecdf(self, x):
        if x <= self.left_bound or x > self.right_bound:
            return None
        if not self.left and not self.right:
            return self.treecdf_A(x)
        else:
            x_left, x_right = self.left.treecdf(x), self.right.treecdf(x)
            if x_left:
                x_new = x_left
            elif x_right:
                x_new = x_right
            else:
                return None
            return self.treecdf_A(x_new)
#    def scalar_density(self, x):
#        if not self.left and not self.right:
#            return 1
#        else:
#            if x <= self.split:
#                return self.pleft * self.left.scalar_density(x)
#            else:
#                return (1 - self.pleft) * self.right.scalar_density(x)
    
    def log_density_univariate(self, y):
        """Calculate the log likelihood of y given the covariate-dependent tree and the splitting probabilities

        Args:
            y (float): univariate y
        """
        if not self:
            return None
        elif y > self.right_bound or y <= self.left_bound:
            return np.NINF
        elif not self.split and not self.left and not self.right: # leaf
            return -np.log(self.right_bound - self.left_bound)
        else:
            if y <= self.split:
                return np.log(self.pleft) + self.left.log_density_univariate(y)
            else:
                return np.log(1 - self.pleft) + self.right.log_density_univariate(y)

            
    def getattr_preorder(self, attr, output):
        """Return node-level attributes in pre-order (DFS)

        Args:
            attr (str): attribute name
            output (List, optional): list containing the attribute values. Defaults to None.
        """
        if self:
            output.append(getattr(self, attr))
            if self.left:
                self.left.getattr_preorder(attr, output)
            if self.right:
                self.right.getattr_preorder(attr, output)
        return output 
    
    def getattr_bfs(self, attr, output):
        """Return node-level attributes in level traversal (BFS)

        Args:
            attr (str): attribute name
            output (List, optional): list containing the attribute values. Defaults to None.
        """
        if not self:
            return []
        else:
            queue = [self]
            while queue:
                node = queue.pop(0)
                output.append(getattr(node, attr))
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
    
    def setattr_bfs_iterative(self, attr, input):
        """Set node-level attributes in level traversal (BFS)

        Args:
            attr (str): attribute name
            input (List): values of the node-level attribute. 
        """
        if not self:
            return None
        else:
            queue = [self]
            while queue:
                node = queue.pop(0)
                setattr(node, attr, input.pop(0))
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
    
    def split2bound(self):
        """Calculate and set the left and right bounds of nodes from splits of internal nodes via top-down recursion. 
        The left and right bound of the root node is assumed as known and already set. 
        """
        if not self: 
            pass
        elif not self.left and not self.right and not self.split: # leaf
            pass
        else:
            # set bounds of left and right children
            self.left.left_bound = self.left_bound
            self.left.right_bound = self.split
            self.right.left_bound = self.split
            self.right.right_bound = self.right_bound
            # recursion
            self.left.split2bound()
            self.right.split2bound()

    def generate_full(self, depth, root):
        """Generate a full binary tree with given depth

        Args:
            depth (int): depth of the binary tree
            root (TreeNode): the root node to start with
        """
        if depth == 1:
            return None
        else:
            root.left = TreeNode()
            root.right = TreeNode()
            self.generate_full(depth - 1, root.left)
            self.generate_full(depth - 1, root.right)
     
    def fit_logistic_middle(self, residual, X, max_depth):
        """fit logistic regression on a rooted full binary tree

        Args:
            self (TreeNode): the root node
            residual (Array): current residuals
            X (Array): covariates X
            max_depth (int): maximum depth of the tree
        """
        curr_depth = 1
        # level traversal
        queue = [self]
        while curr_depth <= max_depth and queue:
            l = len(queue)
            for i in range(l):
                node = queue.pop(0)
                if np.sum(np.logical_and(residual > node.left_bound, residual <= node.right_bound)) == 0:
                    node.pleft = 0.5
                elif np.sum(np.logical_and(residual > node.left_bound, residual <= node.split)) == 0:
                    node.pleft = 0.0
                elif np.sum(np.logical_and(residual > node.split, residual <= node.right_bound)) == 0:
                    node.pleft = 1.0
                else:
                    model = LogisticRegression(solver='saga', random_state=0, penalty='none')
                    local_index = np.logical_and(residual > node.left_bound, residual <= node.right_bound)
                    y_local = residual[local_index]
                    x_local = X[local_index]
                    model.fit(x_local.reshape(-1, 1), (y_local > node.split).astype('int'))
                    node.theta = [model.intercept_, model.coef_]
                    # node.pleft = model.predict_proba(x_local)[:,0]
                if curr_depth <= max_depth - 1:
                    node.left = TreeNode(split = (node.left_bound + node.split)/2, left_bound=node.left_bound, right_bound=node.split)
                    node.right = TreeNode(split = (node.right_bound + node.split)/2, left_bound=node.split, right_bound=node.right_bound)
                    queue.append(node.left)
                    queue.append(node.right)
            curr_depth += 1
    
    # def fit_logistic_greedy(self, residual, X, max_depth, n_grid):
    #     """Fit logistic regression on a rooted full binary tree. 
    #     The optimal partition point is obtained by grid search. 

    #     Args:
    #         self (TreeNode): the root node
    #         residual (Array): current residuals
    #         X (Array): covariates X
    #         max_depth (int): maximum depth of the tree
    #         n_grid (int): number of grids for searching for best partition point
    #     """
    #     curr_depth = 1
    #     # level traversal
    #     queue = [self]
    #     while curr_depth <= max_depth and queue:
    #         l = len(queue)
    #         for i in range(l):
    #             node = queue.pop(0)
    #             split_grid = np.linspace(0, 1, n_grid, endpoint=False)[1:]
    #             min_logloss = sys.float_info.max
    #             optimal_split = 0.5
    #             for split in split_grid:
    #                 node.split = split
    #                 if np.sum(np.logical_and(residual > node.left_bound, residual <= node.right_bound)) == 0:
    #                     node.pleft = 0.5
    #                     optimal_split = 0.5
    #                     break
    #                 elif np.sum(np.logical_and(residual > node.left_bound, residual <= node.split)) == 0:
    #                     # node.pleft = 0.0
    #                     pass
    #                 elif np.sum(np.logical_and(residual > node.split, residual <= node.right_bound)) == 0:
    #                     # node.pleft = 1.0
    #                     pass
    #                 else:
    #                     model = LogisticRegression(solver='saga', random_state=0, penalty='none')
    #                     local_index = np.logical_and(residual > node.left_bound, residual <= node.right_bound)
    #                     y_local = residual[local_index]
    #                     x_local = X[local_index]
    #                     y_true = (y_local > node.split).astype('int')
    #                     model.fit(x_local.reshape(-1, 1), y_true)
    #                     node.theta = [model.intercept_, model.coef_]
    #                     # node.pleft = model.predict_proba(x_local)[:,0]
    #                     y_pred = model.predict_proba(x_local.reshape(-1, 1))
    #                     logloss = log_loss(y_true, y_pred)
    #                     if logloss < min_logloss:
    #                         min_logloss = logloss
    #                         optimal_split = split
    #             node.split = optimal_split
    #             if np.sum(np.logical_and(residual > node.left_bound, residual <= node.right_bound)) == 0:
    #                 # NEED TO pass HERE
    #                 node.pleft = 0.5
    #                 node.split = 0.5
    #             else:
    #                 model = LogisticRegression(solver='saga', random_state=0, penalty='none')
    #                 local_index = np.logical_and(residual > node.left_bound, residual <= node.right_bound)
    #                 y_local = residual[local_index]
    #                 x_local = X[local_index]
    #                 model.fit(x_local.reshape(-1, 1), (y_local > node.split).astype('int'))
    #                 node.theta = [model.intercept_, model.coef_]
    #             if curr_depth <= max_depth - 1:
    #                 node.left = TreeNode(split = None, left_bound=node.left_bound, right_bound=node.split)
    #                 node.right = TreeNode(split = None, left_bound=node.split, right_bound=node.right_bound)
    #                 queue.append(node.left)
    #                 queue.append(node.right)
    #         curr_depth += 1            

    def fit_logistic_greedy_regularization(self, residual, X, max_depth, n_grid, regularization):
        """Fit logistic regression on a rooted full binary tree. 
        The optimal partition point is obtained by grid search. 

        Args:
            self (TreeNode): the root node
            residual (Array): current residuals
            X (Array): covariates X
            max_depth (int): maximum depth of the tree
            n_grid (int): number of grids for searching for best partition point
            regularization (function): (log) regularization function on the splitting point, for example, lambda x: (x-1/2)**2 to encourage balanceness
        """
        curr_depth = 1
        # level traversal
        queue = [self]
        while curr_depth <= max_depth and queue:
            l = len(queue)
            for i in range(l):
                node = queue.pop(0)
                split_grid = np.linspace(0, 1, n_grid, endpoint=False)[1:]
                min_logloss = sys.float_info.max
                if np.sum(np.logical_and(residual > node.left_bound, residual <= node.right_bound)) == 0:
                # stop splitting if there is no data
                    node.split = None
                    node.left = None
                    node.right = None
                    node.theta = None
                    node.pleft = None
                else:
                    for split in split_grid:
                        if np.sum(np.logical_and(residual > node.left_bound, residual <= split)) == 0:
                            logloss = regularization(split)
                            if logloss < min_logloss:
                                min_logloss = logloss
                                node.pleft = 0.0
                                node.split = split
                                node.theta = "one-side"
                        elif np.sum(np.logical_and(residual > split, residual <= node.right_bound)) == 0:
                            logloss = regularization(split)
                            if logloss < min_logloss:
                                min_logloss = logloss
                                node.pleft = 1.0
                                node.split = split
                                node.theta = "one-side"
                        else:
                            model = LogisticRegression(solver='saga', random_state=0, penalty='none')
                            local_index = np.logical_and(residual > node.left_bound, residual <= node.right_bound)
                            y_local = residual[local_index]
                            x_local = X[local_index]
                            y_true = (y_local > split).astype('int')
                            model.fit(x_local.reshape(-1, 1), y_true)
                            y_pred = model.predict_proba(x_local.reshape(-1, 1))
                            logloss = log_loss(y_true, y_pred) + regularization(split)
                            if logloss < min_logloss:
                                min_logloss = logloss
                                node.split = split
                                node.theta = [model.intercept_, model.coef_]
                if curr_depth <= max_depth - 1:
                    node.left = TreeNode(split = None, left_bound=node.left_bound, right_bound=node.split)
                    node.right = TreeNode(split = None, left_bound=node.split, right_bound=node.right_bound)
                    queue.append(node.left)
                    queue.append(node.right)
            curr_depth += 1            
    
    def MCsample_recursive(self):
        """Draw 1 Monte Carlo samples from a tree with pleft fully specified via recursion
        Assume the tree is full binary tree. 
        """
        left, right = self.left_bound, self.right_bound
        if not self.left and not self.right:
            return uniform.rvs(loc = left, scale = right - left)
        if self.left and self.right:
            u = uniform.rvs(loc = 0, scale = 1)
            if u <= self.pleft:
                return self.left.MCsample_recursive()
            else:
                return self.right.MCsample_recursive()
        
    def MCsample_univariate(self, size ):
        """Draw Monte Carlo samples from a tree with pleft fully specified in univariate case

        Args:
            size (int): number of MC samples
        """
        return None



def make_copy(from_tree: TreeNode, to_tree: TreeNode):
        if not from_tree:
            return None
        to_tree.split = from_tree.split
        to_tree.pleft = from_tree.pleft
        to_tree.left_bound = from_tree.left_bound
        to_tree.right_bound = from_tree.right_bound
        to_tree.theta = from_tree.theta
        if from_tree.left:
            to_tree.left = TreeNode()
            # to_tree.left = make_copy(from_tree.left, to_tree.left)
            make_copy(from_tree.left, to_tree.left)
        if from_tree.right:
            to_tree.right = TreeNode()
            # to_tree.right = make_copy(from_tree.right, to_tree.right)
            make_copy(from_tree.right, to_tree.right)
        # return to_tree

def lr2prob(node, x):
    """Calculate the branching probabilities from coefficients of Logistic regression and assign to nodes
    Args:
        node (TreeNode): tree
        x: estimated parameters from Logistic regression
    """
    if not node or not node.theta:
        return None
    else:
        node.pleft = 1.0 / (1 + np.exp(node.theta[0][0] + x * node.theta[1][0][0]))
        lr2prob(node.left, x)
        lr2prob(node.right, x)