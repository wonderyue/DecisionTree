import math
import numpy as np
from collections import Counter
from enum import Enum


class DecisionTree(object):
    """
    DecisionTree
    """

    Mode = Enum("Mode", ("Entropy", "Variance"))
    ENTROPY_THRESHOLD = 1e-6

    class Node(object):
        """
        DecisionTree Node
        """

        def __init__(self, attribute, sample, majority, is_leaf, depth):
            self.attribute = attribute
            self.sample = sample
            self.majority = majority
            self.is_leaf = is_leaf
            self.depth = depth
            self.children = {}  # attribute value : child node

        def clone(self):
            node = DecisionTree.Node(
                self.attribute, self.sample, self.majority, self.is_leaf, self.depth
            )
            for key in self.children:
                node.children[key] = self.children[key].clone()
            return node

    def __init__(self, mode):
        self.root = None
        self.mode = mode

    def clone(self):
        tree = DecisionTree(self.mode)
        tree.root = self.root.clone()
        return tree

    def add_node(self, data, attribute, parents, depth):
        if attribute is None:
            return DecisionTree.Node(
                attribute, len(data), DecisionTree.get_majority(data, -1), True, depth,
            )
        parents.add(attribute)
        node = DecisionTree.Node(
            attribute, len(data), DecisionTree.get_majority(data, -1), False, depth,
        )
        total = len(data)
        counts = [0, 0]
        for row in data:
            counts[row[attribute]] += 1
        for key, count in enumerate(counts):
            if count > 0:
                sub_data = np.array([row for row in data if row[attribute] == key])
                # leaf node: run out of attributes or entropy is lower than threshold
                if (
                    len(parents) == len(sub_data[0]) - 1
                    or self.entropy(sub_data) < DecisionTree.ENTROPY_THRESHOLD
                ):
                    node.children[key] = self.add_node(
                        sub_data, None, parents, depth + 1
                    )
                    continue
                # internal node
                child_attr = self.find_attribute(sub_data, parents)
                node.children[key] = self.add_node(
                    sub_data, child_attr, parents, depth + 1
                )
            else:
                node.children[key] = self.add_node(data, None, parents, depth + 1)
        return node

    def train(self, data):
        attr = self.find_attribute(data, {})
        self.root = self.add_node(data, attr, {attr}, 0)

    def entropy(self, data):
        h = 0
        total = len(data)
        counts = [0, 0]
        for row in data:
            counts[row[-1]] += 1
        if counts[0] == 0 or counts[1] == 0:
            return 0
        for key, count in enumerate(counts):
            p = count / total
            h -= p * math.log2(p)
        return h

    def variance(self, data):
        total = len(data)
        count0 = 0
        for row in data:
            if row[-1] == 0:
                count0 += 1
        return count0 * (total - count0) / (total * total)

    def gain(self, data, attribute):
        h = 0  # sum of P(v)*(H(v) or V(v))
        total = len(data)
        counts = [0, 0]
        for row in data:
            counts[row[attribute]] += 1
        for key, count in enumerate(counts):
            if count > 0:
                d = np.array([row for row in data if row[attribute] == key])
                h += (
                    count
                    / total
                    * (
                        self.entropy(d)
                        if self.mode == DecisionTree.Mode.Entropy
                        else self.variance(d)
                    )
                )
        return h

    def find_attribute(self, data, parents):
        if len(data) == 0:
            return None
        max_attr = None
        min_h = 1
        for attr in range(0, len(data[0]) - 1):
            if attr in parents:
                continue
            h = self.gain(data, attr)
            if h < min_h:
                min_h = h
                max_attr = attr
        return max_attr

    def test(self, data):
        accurate = 0
        for row in data:
            node = self.root
            while not node.is_leaf:
                v = row[node.attribute]
                node = node.children[v]
            if node.majority == row[-1]:
                accurate += 1
        return accurate / len(data)

    def find_max_reduced_error_node(self, node, data, max_accuracy):
        if node.is_leaf:
            return max_accuracy, None
        node.is_leaf = True
        accuracy = self.test(data)
        max_node = None
        if accuracy > max_accuracy:
            max_node = node
            max_accuracy = accuracy
        node.is_leaf = False
        for key in node.children:
            accu, child = self.find_max_reduced_error_node(
                node.children[key], data, max_accuracy
            )
            if accu > max_accuracy:
                max_node = child
                max_accuracy = accu
        return max_accuracy, max_node

    def reduced_error_prune(self, data):
        accuracy = self.test(data)
        max_accuracy, node = self.find_max_reduced_error_node(self.root, data, accuracy)
        while node is not None:
            node.is_leaf = True
            accuracy = self.test(data)
            max_accuracy, node = self.find_max_reduced_error_node(
                self.root, data, accuracy
            )

    def depth_based_prune_dfs(self, node, depth):
        if node is None:
            return
        if node.depth >= depth:
            node.is_leaf = True
        else:
            for key in node.children:
                self.depth_based_prune_dfs(node.children[key], depth)

    def depth_based_prune(self, data, depth_arr):
        depth_arr.sort(reverse=True)
        best_depth = 0
        max = self.test(data)
        copy = self.clone()
        for depth in depth_arr:
            copy.depth_based_prune_dfs(copy.root, depth)
            accuracy = copy.test(data)
            if accuracy > max:
                max = accuracy
                best_depth = depth
        if best_depth > 0:
            self.depth_based_prune_dfs(self.root, best_depth)
        return best_depth

    @staticmethod
    def get_majority(data, attribute):
        count0 = 0
        for row in data:
            if row[attribute] == 0:
                count0 += 1
        return 0 if count0 > len(data) / 2 else 1

    @staticmethod
    def getBasicInfo(node, map):
        if node is None:
            return
        map["total"] = map.get("total", 0) + 1
        depth = [100, 50, 20, 15, 10, 5]
        if node.is_leaf:
            for d in depth:
                if node.depth >= d:
                    map[f"depth>={d}"] = map.get(d, 0) + 1
        else:
            for key in node.children:
                DecisionTree.getBasicInfo(node.children[key], map)
        return map
