"""BEAD: distilling biological age gaps from environmental information.

For each organ and modality (imaging first, then protein):
    biology -> sex-specific age teacher -> age-bias-corrected teacher BAG
    environmental information -> Ridge/LightGBM anchor + neural correction
    biology -> training-only modules -> auxiliary supervision of the encoder

The student predicts:
    ed-BAG = target_mean + target_sd * ((1-w)*Ridge(x) + w*LightGBM(x) + delta(x))
where the three prediction terms are trained in standardized teacher-BAG units.
The loss is MSE(student, teacher) + 0.1*masked_module_MSE + 0.02*mean(delta**2).

Dependencies: numpy, scipy, scikit-learn, lightgbm, torch.
Run a small synthetic-data example with: python BEAD.py --demo

API
---
BEADStudent.fit() accepts healthy training participants and teacher BAG targets.
BEADStudent.predict() requires environmental information only; age and sex are
passed as separate arrays to apply the registered input encoding consistently.
Environmental information is the study's broad input definition, including
lifestyle, psychosocial, socioeconomic, anthropometric and routine measures.
Categorical variables must already follow the registered numerical/one-hot
codebook. Imaging/protein measurements are never student prediction inputs.

fit_bead_oof() demonstrates the five-fold teacher/student training workflow.
Its teacher_factory(sex_code, seed) must return a NEW unfitted sklearn-style
estimator or Pipeline with the prespecified organ- and sex-specific teacher
family, including its own training-fitted biological preprocessing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torch import nn


@dataclass(frozen=True)
class BEADConfig:
    ridge_alphas: tuple[float, ...] = (100.0, 1000.0, 5000.0)
    tree_leaves: tuple[int, ...] = (15, 31)
    tree_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    learning_rates: tuple[float, ...] = (0.0003, 0.001)
    max_trees: int = 500
    tree_patience: int = 25
    max_epochs: int = 50
    neural_patience: int = 8
    batch_size: int = 512
    auxiliary_weight: float = 0.1
    correction_penalty: float = 0.02
    selection_seed: int = 963
    fit_seed: int = 1963


def participant_folds(ids, n=5, person=b"bag99outer"):
    """Keep every occurrence of a participant in the same deterministic fold."""
    return np.asarray([
        int.from_bytes(hashlib.blake2b(str(i).encode("utf-8"), digest_size=8,
                                      person=person).digest(), "little") % n
        for i in ids
    ], dtype=int)


def _numeric_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Expected a nonempty two-dimensional feature matrix.")
    return np.where(np.isfinite(values), values, np.nan)


def _covariates(age, sex, age_mean):
    a = (np.asarray(age) - age_mean) / 10.0
    return np.column_stack([np.ones(len(a)), a, a*a, sex, sex*a, sex*a*a])


# 1. Environmental information: training-fitted scaling and observation flags.
class EnvironmentProcessor:
    def fit(self, environment, age, sex, group_ids=None):
        x = _numeric_matrix(environment)
        self.n_features_ = x.shape[1]
        groups = np.arange(x.shape[1]) if group_ids is None else np.asarray(group_ids)
        if groups.shape != (x.shape[1],):
            raise ValueError("group_ids must identify the original field of each column.")
        _, self.groups_ = np.unique(groups, return_inverse=True)
        self.n_groups_ = int(self.groups_.max()) + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nan_to_num(np.nanmean(x, axis=0))
            scale = np.nanstd(x, axis=0)
        self.scale_ = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        self.age_mean_ = float(np.mean(age))
        self.age_scale_ = max(float(np.std(age)), 1.0)
        matrix = self._matrix(x, age, sex)
        self.keep_ = np.flatnonzero(matrix.std(axis=0) > 1e-6)
        if not len(self.keep_):
            raise ValueError("No variable environmental inputs remain in the training set.")
        # PLE is applied to continuous environmental columns (>16 observed values).
        self.continuous_ = np.asarray([
            j for j, original in enumerate(self.keep_)
            if original < x.shape[1]
            and len(np.unique(x[np.isfinite(x[:, original]), original])) > 16
        ], dtype=int)
        self.knots_ = np.quantile(matrix[:, self.keep_][:, self.continuous_],
                                  np.linspace(0, 1, 9), axis=0).T.astype(np.float32)
        return self

    def _matrix(self, environment, age, sex):
        x = _numeric_matrix(environment)
        age, sex = np.asarray(age, float), np.asarray(sex, float)
        if x.shape[1] != self.n_features_ or age.shape != (len(x),) or sex.shape != (len(x),):
            raise ValueError("Input dimensions differ from the registered feature layout.")
        if not np.isfinite(age).all() or not np.isin(sex, [0, 1]).all():
            raise ValueError("Age must be finite and sex must use the registered 0/1 coding.")
        observed = np.isfinite(x)
        z = np.clip(np.where(observed, (x-self.mean_)/self.scale_, 0.0), -12, 12)
        flags = np.column_stack([
            observed[:, self.groups_ == g].any(axis=1) for g in range(self.n_groups_)
        ])
        return np.column_stack([z, flags, (age-self.age_mean_)/self.age_scale_,
                                sex, flags.mean(axis=1)]).astype(np.float32)

    def transform(self, environment, age, sex):
        return self._matrix(environment, age, sex)[:, self.keep_]


# 2. Privileged biological targets: fit on the current healthy training subset.
class BiologicalModules:
    def fit(self, biology, age, sex, modality="imaging", categories=None, seed=963):
        raw = _numeric_matrix(biology)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eligible = (np.isfinite(raw).mean(axis=0) >= 0.90) & (np.nanstd(raw, axis=0) > 1e-6)
        self.columns_ = np.flatnonzero(eligible)
        if not len(self.columns_):
            raise ValueError("No biological feature meets the 90% observation threshold.")
        raw = raw[:, self.columns_]
        self.medians_ = np.nanmedian(raw, axis=0)
        filled = np.where(np.isfinite(raw), raw, self.medians_)
        self.age_mean_ = float(np.mean(age))
        covariates = _covariates(age, sex, self.age_mean_)
        self.coefficients_ = np.linalg.lstsq(covariates, filled, rcond=None)[0]
        residual = filled - covariates @ self.coefficients_
        self.scale_ = np.maximum(residual.std(axis=0), 1e-6)
        residual = np.clip(residual/self.scale_, -12, 12)
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(raw), min(len(raw), 5000), replace=False)
        p = raw.shape[1]
        category_labels = None
        if categories is not None and modality == "imaging":
            categories = np.asarray(categories)
            if categories.shape != (biology.shape[1],):
                raise ValueError("One predefined category is required per biological column.")
            _, category_labels = np.unique(categories[self.columns_], return_inverse=True)
        if p <= 8:
            self.labels_ = np.arange(p)
        elif category_labels is not None and 2 <= len(np.unique(category_labels)) <= 8:
            self.labels_ = category_labels
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                correlations = np.corrcoef(residual[sample], rowvar=False)
            distance = np.clip(1-np.abs(np.nan_to_num(correlations)), 0, 1)
            np.fill_diagonal(distance, 0)
            clusters = fcluster(linkage(squareform(distance, checks=False), method="average"),
                                t=8, criterion="maxclust")
            _, self.labels_ = np.unique(clusters, return_inverse=True)
        self.n_modules_ = int(self.labels_.max()) + 1
        self.loadings_ = np.zeros((p, self.n_modules_))
        for module in range(self.n_modules_):
            columns = np.flatnonzero(self.labels_ == module)
            if len(columns) == 1:
                self.loadings_[columns, module] = 1.0
            else:
                pc = PCA(n_components=1, svd_solver="randomized", random_state=seed+module)
                vector = pc.fit(residual[sample][:, columns]).components_[0]
                vector *= np.sign(vector[np.argmax(np.abs(vector))])
                self.loadings_[columns, module] = vector
        scores = residual @ self.loadings_
        self.mean_ = scores.mean(axis=0)
        self.module_scale_ = np.maximum(scores.std(axis=0), 1e-6)
        return self

    def transform(self, biology, age, sex):
        raw = _numeric_matrix(biology)[:, self.columns_]
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.medians_)
        residual = filled - _covariates(age, sex, self.age_mean_) @ self.coefficients_
        residual = np.clip(residual/self.scale_, -12, 12)
        scores = (residual @ self.loadings_-self.mean_)/self.module_scale_
        mask = np.column_stack([
            observed[:, self.labels_ == j].mean(axis=1) >= 0.80
            for j in range(self.n_modules_)
        ])
        return scores.astype(np.float32), mask.astype(np.float32)


# 3. Linear/tree anchor; its cross-predictions train the neural correction.
def _tree(leaves, iterations, seed):
    return lgb.LGBMRegressor(
        objective="regression", n_estimators=iterations, learning_rate=0.03,
        num_leaves=leaves, max_depth=5 if leaves == 15 else -1,
        min_child_samples=100, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0,
        random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def _select_anchor(x, y, xv, yv, config):
    best_ridge = None
    for alpha in config.ridge_alphas:
        ridge = Ridge(alpha=alpha, solver="lsqr").fit(x, y)
        score = float(np.mean((ridge.predict(xv)-yv)**2))
        if best_ridge is None or score < best_ridge[0]:
            best_ridge = (score, alpha, ridge)
    best_tree = None
    for leaves in config.tree_leaves:
        tree = _tree(leaves, config.max_trees, config.selection_seed)
        tree.fit(x, y, eval_set=[(xv, yv)], eval_metric="l2",
                 callbacks=[lgb.early_stopping(config.tree_patience, verbose=False)])
        score = float(np.mean((tree.predict(xv)-yv)**2))
        if best_tree is None or score < best_tree[0]:
            best_tree = (score, leaves, tree)
    ridge, tree = best_ridge[2], best_tree[2]
    r, t = ridge.predict(xv), tree.predict(xv)
    weight = min(config.tree_weights, key=lambda w: np.mean(((1-w)*r+w*t-yv)**2))
    return dict(alpha=best_ridge[1], leaves=best_tree[1],
                iterations=int(tree.best_iteration_ or config.max_trees), tree_weight=weight)


def _fit_anchor(x, y, ids, settings, seed):
    ridge = Ridge(alpha=settings["alpha"], solver="lsqr").fit(x, y)
    tree = _tree(settings["leaves"], settings["iterations"], seed).fit(x, y)
    folds = participant_folds(ids, n=3, person=b"bag99teachin")
    cross = np.full(len(y), np.nan, dtype=np.float32)
    weight = settings["tree_weight"]
    for fold in range(3):
        valid = folds == fold
        train = ~valid
        if not valid.any() or train.sum() < 2:
            raise ValueError("Each anchor cross-fitting fold needs training and validation rows.")
        r = Ridge(alpha=settings["alpha"], solver="lsqr").fit(x[train], y[train])
        t = _tree(settings["leaves"], settings["iterations"], seed+fold).fit(x[train], y[train])
        cross[valid] = (1-weight)*r.predict(x[valid]) + weight*t.predict(x[valid])
    return ridge, tree, cross


def _anchor_predict(ridge, tree, settings, x):
    weight = settings["tree_weight"]
    return ((1-weight)*ridge.predict(x)+weight*tree.predict(x)).astype(np.float32)


# 4. Shared environmental encoder with separate BAG and biological-module heads.
class BEADNetwork(nn.Module):
    def __init__(self, input_dim, n_modules, continuous, knots):
        super().__init__()
        self.register_buffer("continuous", torch.as_tensor(continuous, dtype=torch.long))
        self.register_buffer("knots", torch.as_tensor(knots, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim+8*len(continuous), 64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.bag_head = nn.Linear(32, 1)
        self.module_head = nn.Linear(32, n_modules)
        nn.init.zeros_(self.bag_head.weight)
        nn.init.zeros_(self.bag_head.bias)

    def forward(self, x, anchor):
        values = x[:, self.continuous, None]
        segments = ((values-self.knots[None, :, :-1]) /
                    (self.knots[None, :, 1:]-self.knots[None, :, :-1]).clamp_min(0.001))
        h = self.encoder(torch.cat([x, segments.clamp(0, 1).flatten(1)], dim=1))
        delta = 0.25*self.bag_head(h).squeeze(1)
        return anchor+delta, self.module_head(h), delta


def bead_loss(prediction, target, modules, module_target, mask, delta, config):
    """Average observed module errors per participant, then across the minibatch."""
    biological = (((modules-module_target).square()*mask).sum(dim=1) /
                  mask.sum(dim=1).clamp_min(1)).mean()
    return ((prediction-target).square().mean() + config.auxiliary_weight*biological
            + config.correction_penalty*delta.square().mean())


@torch.no_grad()
def _network_predict(network, x, anchor):
    network.eval()
    batches = []
    for start in range(0, len(x), 2048):
        prediction, _, _ = network(torch.as_tensor(x[start:start+2048], dtype=torch.float32),
                                   torch.as_tensor(anchor[start:start+2048], dtype=torch.float32))
        batches.append(prediction.cpu().numpy())
    return np.concatenate(batches) if batches else np.empty(0, dtype=np.float32)


def _fit_network(x, y, z, mask, anchor, processor, config, lr, seed,
                 validation=None, fixed_epochs=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = BEADNetwork(x.shape[1], z.shape[1], processor.continuous_, processor.knots_)
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=0.001)
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (x, y, z, mask, anchor)]
    best_state = copy.deepcopy(network.state_dict())
    selected_epoch, stale = 0, 0
    best_mse = float("inf")
    if validation is not None:
        xv, yv, av = validation
        # Epoch zero retains the anchor if no neural epoch improves validation MSE.
        best_mse = float(np.mean((_network_predict(network, xv, av)-yv)**2))
    epochs = config.max_epochs if fixed_epochs is None else fixed_epochs
    for epoch in range(1, epochs+1):
        network.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), config.batch_size):
            batch = order[start:start+config.batch_size]
            xb, yb, zb, mb, ab = [a[batch] for a in tensors]
            prediction, modules, delta = network(xb, ab)
            loss = bead_loss(prediction, yb, modules, zb, mb, delta, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), max_norm=2.0)
            optimizer.step()
        if validation is None:
            selected_epoch = epoch
            continue
        mse = float(np.mean((_network_predict(network, xv, av)-yv)**2))
        if mse < best_mse-1e-5:
            best_mse, selected_epoch, stale = mse, epoch, 0
            best_state = copy.deepcopy(network.state_dict())
        else:
            stale += 1
        if stale >= config.neural_patience:
            break
    if validation is not None:
        network.load_state_dict(best_state)
    return network, dict(lr=lr, epochs=selected_epoch, validation_mse=best_mse)


# 5. Student: select within healthy training data, then refit with fixed settings.
class BEADStudent:
    def __init__(self, config=None):
        self.config = BEADConfig() if config is None else config

    def fit(self, environment, age, sex, teacher_bag, biology, ids,
            modality="imaging", group_ids=None, categories=None, validation_fold=0):
        """Pass ONLY healthy outer-training rows; teacher_bag must be cross-fitted."""
        environment, biology = _numeric_matrix(environment), _numeric_matrix(biology)
        age, sex, target = [np.asarray(a, float) for a in (age, sex, teacher_bag)]
        ids = np.asarray(ids, dtype=str)
        n = len(ids)
        if any(a.shape != (n,) for a in (age, sex, target)) or len(environment) != n or len(biology) != n:
            raise ValueError("All input arrays must have identical participant order and length.")
        if len(np.unique(ids)) != n or not np.isfinite(target).all():
            raise ValueError("Training IDs must be unique and teacher BAG targets must be finite.")
        if modality not in ("imaging", "protein") or validation_fold not in range(5):
            raise ValueError("Use modality imaging/protein and validation_fold 0..4.")
        if not np.isfinite(age).all() or not np.isin(sex, [0, 1]).all():
            raise ValueError("Age must be finite and sex must use 0/1 coding.")
        split = participant_folds(ids, n=5, person=b"bag99studval")
        valid = split == validation_fold
        train = ~valid
        if min(train.sum(), valid.sum()) < 2:
            raise ValueError("Insufficient healthy participants for the student validation split.")
        config = self.config

        def components(rows, seed):
            ep = EnvironmentProcessor().fit(environment[rows], age[rows], sex[rows], group_ids)
            bm = BiologicalModules().fit(biology[rows], age[rows], sex[rows], modality, categories, seed)
            x = ep.transform(environment, age, sex)
            z, mask = bm.transform(biology, age, sex)
            mean, sd = float(target[rows].mean()), max(float(target[rows].std()), 1e-6)
            y = ((target-mean)/sd).astype(np.float32)
            return ep, bm, x, z, mask, mean, sd, y

        ep, _, x, z, mask, _, _, y = components(train, config.selection_seed)
        settings = _select_anchor(x[train], y[train], x[valid], y[valid], config)
        ridge, tree, cross = _fit_anchor(x[train], y[train], ids[train], settings, config.selection_seed)
        anchor_valid = _anchor_predict(ridge, tree, settings, x[valid])
        selected = None
        for lr in config.learning_rates:
            _, trial = _fit_network(x[train], y[train], z[train], mask[train], cross,
                                    ep, config, lr, config.selection_seed,
                                    validation=(x[valid], y[valid], anchor_valid))
            if selected is None or trial["validation_mse"] < selected["validation_mse"]:
                selected = trial

        # Re-estimate preprocessing, modules and target scaling on all healthy rows.
        all_rows = np.ones(n, dtype=bool)
        self.environment_, self.modules_, x, z, mask, self.target_mean_, self.target_sd_, y = components(all_rows, config.fit_seed)
        self.ridge_, self.tree_, cross = _fit_anchor(x, y, ids, settings, config.fit_seed)
        self.network_, _ = _fit_network(x, y, z, mask, cross, self.environment_, config,
                                       selected["lr"], config.fit_seed, fixed_epochs=selected["epochs"])
        self.anchor_settings_, self.neural_settings_ = settings, selected
        self.modality_ = modality
        return self

    def predict(self, environment, age, sex):
        """Return ed-BAG in years; no measured imaging/protein data are required."""
        if not hasattr(self, "network_"):
            raise RuntimeError("Call fit before predict.")
        x = self.environment_.transform(environment, age, sex)
        anchor = _anchor_predict(self.ridge_, self.tree_, self.anchor_settings_, x)
        prediction = _network_predict(self.network_, x, anchor)
        return prediction*self.target_sd_ + self.target_mean_


# 6. Teacher targets and participant-level outer validation.
def teacher_targets(biology, age, sex, ids, train, held_out,
                    teacher_factory: Callable[[int, int], Any], seed=963):
    """Generate three-fold teacher labels and sex-specific quadratic age correction.

    train contains healthy outer-training participants; held_out contains the
    outer test fold. The factory must fit all biological preprocessing inside
    each returned estimator/Pipeline. No full-cohort fitted teacher is accepted.
    """
    inner = participant_folds(ids, n=3, person=b"bag99teachin")
    target = np.full(len(ids), np.nan)
    for code in (0, 1):
        train_sex = train & (sex == code)
        test_sex = held_out & (sex == code)
        cross_age = np.full(len(ids), np.nan)
        for fold in range(3):
            fit_rows = train_sex & (inner != fold)
            val_rows = train_sex & (inner == fold)
            if fit_rows.sum() < 3 or not val_rows.any():
                raise ValueError("Insufficient healthy participants in a sex-specific teacher fold.")
            teacher = teacher_factory(code, seed+10*code+fold)
            teacher.fit(biology[fit_rows], age[fit_rows])
            cross_age[val_rows] = np.asarray(teacher.predict(biology[val_rows])).reshape(-1)
        age_mean = float(age[train_sex].mean())
        a = (age-age_mean)/10.0
        design = np.column_stack([np.ones(len(age)), a, a*a])
        bias = np.linalg.lstsq(design[train_sex], cross_age[train_sex]-age[train_sex], rcond=None)[0]
        target[train_sex] = cross_age[train_sex]-age[train_sex]-design[train_sex] @ bias
        if test_sex.any():
            teacher = teacher_factory(code, seed+10*code+9)
            teacher.fit(biology[train_sex], age[train_sex])
            predicted_age = np.asarray(teacher.predict(biology[test_sex])).reshape(-1)
            target[test_sex] = predicted_age-age[test_sex]-design[test_sex] @ bias
    if not np.isfinite(target[train | held_out]).all():
        raise ValueError("Teacher predictions must be finite for all fitted/evaluated rows.")
    return target


def fit_bead_oof(environment, biology, age, sex, ids, healthy, teacher_factory,
                 modality="imaging", group_ids=None, categories=None, config=None):
    """Five outer folds for ONE organ/modality; return OOF scores and fitted students.

    Apply registered eligibility and missingness rules before this function.
    All arrays must be aligned to unique participant IDs, encoded consistently
    across systems. Call imaging systems before protein systems. For a whole-
    modality target, supply its pooled biological panel and its own teacher;
    do not substitute the average of organ-specific scores.
    """
    config = BEADConfig() if config is None else config
    environment, biology = _numeric_matrix(environment), _numeric_matrix(biology)
    ids = np.asarray(ids, dtype=str)
    age, sex = np.asarray(age, float), np.asarray(sex, float)
    healthy = np.asarray(healthy)
    n = len(ids)
    if healthy.shape != (n,) or not np.isin(healthy, [0, 1]).all():
        raise ValueError("healthy must be a complete Boolean/0-1 vector.")
    healthy = healthy.astype(bool)
    if len(np.unique(ids)) != n or any(a.shape != (n,) for a in (age, sex)):
        raise ValueError("Participant IDs must be unique and demographic arrays aligned.")
    if len(environment) != n or len(biology) != n or not np.isfinite(age).all() or not np.isin(sex, [0, 1]).all():
        raise ValueError("Check matrix alignment, finite age and registered sex coding.")
    folds = participant_folds(ids)
    prediction, target = np.full(n, np.nan), np.full(n, np.nan)
    models = []
    for fold in range(5):
        train, test = healthy & (folds != fold), folds == fold
        if not test.any():
            raise ValueError("Every outer fold must contain participants.")
        labels = teacher_targets(biology, age, sex, ids, train, test, teacher_factory,
                                 seed=config.selection_seed+100*fold)
        student = BEADStudent(config).fit(
            environment[train], age[train], sex[train], labels[train], biology[train], ids[train],
            modality=modality, group_ids=group_ids, categories=categories, validation_fold=fold,
        )
        prediction[test] = student.predict(environment[test], age[test], sex[test])
        target[test] = labels[test]
        models.append(student)
    return dict(ed_bag=prediction, teacher_bag=target, outer_fold=folds, models=models)


def demo():
    """Exercise the student API with synthetic data and a small training budget."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(1963)
    n = 600
    environment = rng.normal(size=(n, 12))
    age = rng.uniform(40, 75, n)
    sex = rng.integers(0, 2, n).astype(float)
    latent = environment[:, :5] @ rng.normal(size=(5, 10))
    biology = latent+0.1*age[:, None]+rng.normal(size=(n, 10))
    target = 0.9*environment[:, 0]-0.5*environment[:, 1]+rng.normal(size=n)
    environment[rng.random(environment.shape) < 0.05] = np.nan
    biology[rng.random(biology.shape) < 0.04] = np.nan
    ids = np.asarray([str(i) for i in range(n)])
    train = participant_folds(ids) != 0
    config = BEADConfig(ridge_alphas=(100.0,), tree_leaves=(15,), max_trees=30,
                        tree_patience=5, learning_rates=(0.001,), max_epochs=3,
                        neural_patience=2, batch_size=128)
    model = BEADStudent(config).fit(environment[train], age[train], sex[train],
                                    target[train], biology[train], ids[train])
    prediction = model.predict(environment[~train], age[~train], sex[~train])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Synthetic example produced non-finite predictions.")
    print(f"Synthetic example: {train.sum()} training rows, {len(prediction)} predictions.")
    print("Anchor settings:", model.anchor_settings_)
    print("Neural settings:", model.neural_settings_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAD environmental-information distillation")
    parser.add_argument("--demo", action="store_true", help="Run a small synthetic-data example")
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        parser.print_help()
