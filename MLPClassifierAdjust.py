# this needs to be copied to sklearn_porter\estimator\classifier\MLPClassifier\__init__.py
# instead of the export function so that we write the weights and bias to a file

def export(self, class_name, method_name, export_data=False,
           export_dir='.', export_filename='data.json',
           export_append_checksum=False, **kwargs):
    """
    Port a trained estimator to the syntax of a chosen programming language.

    Parameters
    ----------
    :param class_name : string
        The name of the class in the returned result.
    :param method_name : string
        The name of the method in the returned result.
    :param export_data : bool, default: False
        Whether the model data should be saved or not.
    :param export_dir : string, default: '.' (current directory)
        The directory where the model data should be saved.
    :param export_filename : string, default: 'data.json'
        The filename of the exported model data.
    :param export_append_checksum : bool, default: False
        Whether to append the checksum to the filename or not.

    Returns
    -------
    :return : string
        The transpiled algorithm with the defined placeholders.
    """
    # Arguments:
    self.class_name = class_name
    self.method_name = method_name

    # Estimator:
    est = self.estimator

    self.output_activation = est.out_activation_
    self.hidden_activation = est.activation

    self.n_layers = est.n_layers_
    self.n_hidden_layers = est.n_layers_ - 2

    self.n_inputs = len(est.coefs_[0])
    self.n_outputs = est.n_outputs_

    self.hidden_layer_sizes = est.hidden_layer_sizes
    if isinstance(self.hidden_layer_sizes, int):
        self.hidden_layer_sizes = [self.hidden_layer_sizes]
    self.hidden_layer_sizes = list(self.hidden_layer_sizes)

    self.layer_units = \
        [self.n_inputs] + self.hidden_layer_sizes + [est.n_outputs_]

    # Weights:
    self.coefficients = est.coefs_

    with open('MLPWeights.txt', 'w') as f:
        f.write('[')
        json.dump(self.coefficients[0].tolist(), f)
        f.write(',')
        json.dump(self.coefficients[1].tolist(), f)
        f.write(']')

    # Bias:
    self.intercepts = est.intercepts_

    with open('MLPbias.txt', 'w') as f:
        f.write('[')
        json.dump(self.intercepts[0].tolist(), f)
        f.write(',')
        json.dump(self.intercepts[1].tolist(), f)
        f.write(']')

    # Binary or multiclass classifier?
    self.is_binary = self.n_outputs == 1
    self.prefix = 'binary' if self.is_binary else 'multi'

    if self.target_method == 'predict':
        # Exported:
        if export_data and os.path.isdir(export_dir):
            self.export_data(export_dir, export_filename,
                             export_append_checksum)
            return self.predict('exported')
        # Separated:
        return self.predict('separated')