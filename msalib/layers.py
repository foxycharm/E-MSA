import tensorflow as tf
Base = tf.layers 


class MSALayer(Base.Layer):
    """Родительский класс для Layers, используемый 
    для алгоритмов на основе MSA. Все дополнительные 
    подписи и атрибуты будут начаться с msa_
    https://www.tensorflow.org/api_docs/python/tf/layers/Layer
    """
    def __init__(self, *args, msa_rho=0.1, msa_reg=0.1,
                 msa_trainable=True, **kwargs):
        """msa_rho - вектор сопряженного состояния, msa_reg - 
        """
        super().__init__(*args, **kwargs)
        self.msa_rho = tf.placeholder_with_default(
            msa_rho, [], 'msa_rho')
        """https://www.tensorflow.org/api_docs/python/tf/add_to_collection
        """
        tf.add_to_collection('msa_rho', self.msa_rho)
        self.msa_reg = msa_reg
        self.msa_trainable = msa_trainable

    def msa_regularizer(self):
        if self.variables:
            return self.msa_reg * tf.add_n([
                tf.nn.l2_loss(v) for v in self.variables])
        else:
            return 0.0

    def msa_hamiltonian(self, x, p):
        """𝐻(𝑡, 𝑥, 𝒑, 𝜽) = 〈𝒑, 𝒇(𝑡, 𝒙, 𝜽)〉 
        """
        return tf.reduce_sum(p * self.apply(x))

    def msa_backward(self, x, p):
        """Вычисляет p_{n}

        Аргументы:
            x {tf tensor} -- x_{n}
            p {tf tensor} -- p_{n+1}

        Возвращает:
            tf tensor -- p_{n}
        """

        x = tf.stop_gradient(x)
        p = tf.stop_gradient(p)
        H = self.msa_hamiltonian(x, p)
        return tf.gradients(H, x)[0]

    def msa_minus_H_aug(self, x, y, p, q):
        """Вычисляет отрицательный дополненный Гамильтониан

        Аргументы:
            x {tf tensor} -- x_{n}
            y {tf tensor} -- x_{n+1}
            p {tf tensor} -- p_{n+1}
            q {tf tensor} -- p_{n}

        Возвращает:
            tf tensor -- отрицательный расширенный Гамильтониан
        """

        x, y, p, q = [tf.stop_gradient(t) for t in [x, y, p, q]]
        dHdp = self.apply(x)
        H = tf.reduce_sum(p * dHdp) - self.msa_regularizer()
        dHdx = tf.gradients(H, x)[0]
        x_feasibility = self.msa_rho * tf.nn.l2_loss(y - dHdp)
        p_feasibility = self.msa_rho * tf.nn.l2_loss(q - dHdx)
        return - H + x_feasibility + p_feasibility


class Dense(Base.Dense, MSALayer):
    """https://www.tensorflow.org/api_docs/python/tf/layers/dense
       Правльнее использовать и переделать с keras.layers.Dense
    """
    pass


class ResidualDense(Dense):
    """Остаточный Dense слой
    """

    def __init__(self, *args, delta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_delta = delta

    def call(self, inputs):
        return inputs + self.msa_delta * super().call(inputs)


class Conv2D(Base.Conv2D, MSALayer):
    pass


class ResidualConv2D(Conv2D):
    """Остаточный сверточный слой 2D
    """

    def __init__(self, *args, delta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_delta = delta

    def call(self, inputs):
        return inputs + self.msa_delta * super().call(inputs)


class Lower(MSALayer):
    """Нижний слой
    """

    def __init__(self, *args, lower_axis=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower_axis = lower_axis
        self.msa_trainable = False

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.lower_axis, keepdims=True)


class Flatten(Base.Flatten, MSALayer):
    """Flatten слой
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_trainable = False


class AveragePooling2D(Base.AveragePooling2D, MSALayer):
    """Объединяющий слой по среднему
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_trainable = False
