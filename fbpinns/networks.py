"""
Defines standard neural network models

Each network class must inherit from the Network base class.
Each network class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""
import jax
import jax.numpy as jnp
from jax import random
from jax import lax



class Network:
    """Base neural network class to be inherited by different neural network classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""
        raise NotImplementedError

    # 该方法是一个类方法（使用 @ staticmethod装饰器），用于初始化类的参数。
    # 参数 * args表示该方法可以接受任意数量的位置参数。
    # 方法返回一个元组，包含两个字典，分别是静态参数（staticparameters）和可训练参数（trainable parameters）的 pytree形式（一种树状结构的数据表示方式）。

    @staticmethod
    def network_fn(params, x):
        """Forward model, for a SINGLE point with shape (xd,)"""
        raise NotImplementedError
    #该方法是一个静态方法，用于定义前向传播模型。
    #参数 params 是包含模型参数的 pytree，x 是输入数据的单个点。
    #抽象方法，需要在子类中实现，因为不同的模型会有不同的前向传播逻辑。

class CustomNetwork(Network):

    @staticmethod
    def init_params(key, layer_sizes, subdomain_id):
        keys = random.split(key, len(layer_sizes)-1)
        if subdomain_id == 3:
            # 子域1的层数设计逻辑
            num_layers = 3
        else:
            # 默认情况下的层数设计逻辑
            num_layers = len(layer_sizes) - 1
            # 只选择 num_layers 个密钥进行使用
        keys = keys[(len(layer_sizes)-num_layers-1):]
        params = [CustomNetwork._random_layer_params(k, m, n)
                  for k, m, n in zip(keys, layer_sizes[:-(len(layer_sizes)-num_layers)], layer_sizes[(len(layer_sizes)-num_layers):])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        w_key, b_key = random.split(key)
        # 计算标准差 v。
        v = jnp.sqrt(1 / m)
        # 使用均匀分布随机初始化权重 w。
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        # 使用均匀分布随机初始化偏差 b。
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w, b

    # @staticmethod
    def network_fn(params, x):
        # 从参数中提取子域的神经网络层参数。
        params = params["trainable"]["network"]["subdomain"]["layers"]
        # 遍历网络层，应用权重、偏差和激活函数（tanh）。
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        # 处理最后一层，不应用激活函数。
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

    # def network_fn(params, x, subdomain_id):
    #     # 从参数中提取子域的神经网络层参数。
    #     params = params["trainable"]["network"]["subdomain"][subdomain_id]["layers"]
    #     # 遍历网络层，应用权重、偏差和激活函数（tanh）。
    #     for w, b in params[:-1]:
    #         x = jnp.dot(w, x) + b
    #         x = jnp.tanh(x)
    #     # 处理最后一层，不应用激活函数。
    #     w, b = params[-1]
    #     x = jnp.dot(w, x) + b
    #     return x

class FCN(Network):

    # @staticmethod
    def init_params(key, layer_sizes):
        keys = random.split(key, len(layer_sizes)-1)
        params = [FCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {'layers': params}
        return {}, trainable_params

    # @staticmethod
    def _random_layer_params(key, m, n):
        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w, b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.tanh(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x
    # def network_fn(params, x, mask):
    #     params1 = params["trainable"]["network"]["subdomain"]["layers"]
    #     def process_special(args):
    #         params, x = args
    #         for w, b in params[:-1]:
    #             w = w[:16, :]
    #             b = b[:16]
    #             x = jnp.dot(w, x) + b
    #             x = jnp.tanh(x)
    #         w, b = params[-1]
    #         w = w[:, :16]
    #         x = jnp.dot(w, x) + b
    #         return x
    #     def process_regular(args):
    #         params, x = args
    #         for w, b in params[:-1]:
    #             x = jnp.dot(w, x) + b
    #             x = jnp.tanh(x)
    #         w, b = params[-1]
    #         x = jnp.dot(w, x) + b
    #         return x
    #     x = lax.cond(mask, process_regular,process_special,  (params1, x))
    #     return x
class AdaptiveFCN(Network):

    @staticmethod
    def init_params(key, layer_sizes):

        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveFCN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(1/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        a = jnp.ones_like(b)
        return w,b,a

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b, a in params[:-1]:
            x = jnp.dot(w, x) + b
            x = a*jnp.tanh(x/a)
        w, b, _ = params[-1]
        x = jnp.dot(w, x) + b
        return x

class SIREN(FCN):

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        return w,b

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w, b in params[:-1]:
            x = jnp.dot(w, x) + b
            x = jnp.sin(x)
        w, b = params[-1]
        x = jnp.dot(w, x) + b
        return x

class AdaptiveSIREN(Network):

    @staticmethod
    def init_params(key, layer_sizes):

        keys = random.split(key, len(layer_sizes)-1)
        params = [AdaptiveSIREN._random_layer_params(k, m, n)
                for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]
        trainable_params = {"layers": params}
        return {}, trainable_params

    @staticmethod
    def _random_layer_params(key, m, n):
        "Create a random layer parameters"

        w_key, b_key = random.split(key)
        v = jnp.sqrt(6/m)
        w = random.uniform(w_key, (n, m), minval=-v, maxval=v)
        b = random.uniform(b_key, (n,), minval=-v, maxval=v)
        c,o = jnp.ones_like(b), jnp.ones_like(b)
        return w,b,c,o

    @staticmethod
    def network_fn(params, x):
        params = params["trainable"]["network"]["subdomain"]["layers"]
        for w,b,c,o in params[:-1]:
            x = jnp.dot(w, x) + b
            x = c*jnp.sin(o*x)
        w,b,_,_ = params[-1]
        x = jnp.dot(w, x) + b
        return x


def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu



if __name__ == "__main__":

    x = jnp.ones(2)
    key = random.PRNGKey(0)
    layer_sizes = [2,16,32,16,1]
    for NN in [FCN, AdaptiveFCN, SIREN, AdaptiveSIREN]:
        network = NN
        ps_ = network.init_params(key, layer_sizes)
        params = {"static":{"network":ps_[0]}, "trainable":{"network":{"subdomain":ps_[1]}}}
        print(x.shape, network.network_fn(params, x).shape, NN.__name__)
