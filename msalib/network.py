import tensorflow as tf


class Network(tf.keras.Sequential):
    """Класс нейронной сети, объединяющей слои MSA
       https://www.tensorflow.org/api_docs/python/tf/keras/models/Sequential
    """

    def msa_compute_x(self, input):
        """Вычисление x

        Arguments:
            input {tf tensor} -- 
        """

        super().apply(input)
        self.msa_xs = [l.input for l in self.layers] + [self.output, ]

    def _msa_add_terminal_loss(self, label, loss_func):
        self.label = label
        self.msa_terminal_loss = loss_func(
            self.output, self.label)

    def msa_compute_p(self, label, loss_func):
        """Solve p

        Аргументы:
            label {tf tensor} -- labels
            loss_func {function returning tf tensor} -- функция потерь
        """

        self._msa_add_terminal_loss(label, loss_func)
        p = - tf.gradients(self.msa_terminal_loss, self.output)[0]
        self.msa_ps = [p]
        for layer in reversed(self.layers):
            p = layer.msa_backward(layer.input, p)
            self.msa_ps.append(p)
        self.msa_ps.reverse()
