Code Structure
==============

Models are defined by a :class:`~bayes_covernet.tf.model.abstract_models.ModelFactory`, which is an abstract base class. Realizations usually consist of an base model
(like :class:`~bayes_covernet.tf.model.models.Covernet`) and a probabilistic modeling Mixin (like :class:`~bayes_covernet.tf.model.abstract_models.VIMixin`), resulting in
a complete ModelFactory (:class:`~bayes_covernet.tf.model.models.CovernetVI`). 

Executing a model uses the :class:`~bayes_covernet.tf.model.training.Trainer`, which creates metrics, runs the training and the evaluation. :class:`~bayes_covernet.tf.model.training.TuneableTrainer` is a wrapper
for use with `ray tune <https://docs.ray.io/en/latest/tune/index.html>`_ (see program arguments).