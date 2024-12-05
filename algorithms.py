# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.autograd as autograd
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

import networks
from loss import *

ALGORITHMS = [
    "ERM",
    "PERM",
    "PERMIW",
    "CORAL",
    "MMD",
    "DANN",
    "WD",
    "KL",
    "KLMeanOnly",
    "MMDBound",
    "PMMD",
    "PMMDBound",
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name)
        )
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class PERM(Algorithm):
    """
    Empirical Risk Minimization (ERM) with probabilistic representation network
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PERM, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        self.featurizer = networks.Featurizer(
            input_shape, self.hparams, probabilistic=True
        )
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )

        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters())
            + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.num_samples = hparams["num_samples"]

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        all_z_params = self.featurizer(all_x)
        z_dim = int(all_z_params.shape[-1] / 2)
        z_mu = all_z_params[:, :z_dim]
        z_sigma = F.softplus(all_z_params[:, z_dim:])

        all_z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
        all_z = all_z_dist.rsample()

        loss = F.cross_entropy(self.classifier(all_z), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        z_params = self.featurizer(x)
        z_dim = int(z_params.shape[-1] / 2)
        z_mu = z_params[:, :z_dim]
        z_sigma = F.softplus(z_params[:, z_dim:])

        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)

        probs = 0.0
        for s in range(self.num_samples):
            z = z_dist.rsample()
            probs += F.softmax(self.classifier(z), 1)
        probs = probs / self.num_samples
        return probs


class KL(Algorithm):
    """
    KL
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(KL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(
            input_shape, self.hparams, probabilistic=True
        )
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )

        cls_lr = (
            100 * self.hparams["lr"]
            if hparams["nonlinear_classifier"]
            else self.hparams["lr"]
        )

        self.optimizer = torch.optim.Adam(
            # list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            [
                {
                    "params": self.featurizer.parameters(),
                    "lr": self.hparams["lr"],
                },
                {"params": self.classifier.parameters(), "lr": cls_lr},
            ],
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.num_samples = hparams["num_samples"]
        self.kl_reg = hparams["kl_reg"]
        self.kl_reg_aux = hparams["kl_reg_aux"]
        self.augment_softmax = hparams["augment_softmax"]

    def update(self, minibatches, unlabeled=None):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        x_target = torch.cat(unlabeled)

        total_x = torch.cat([x, x_target])
        total_z_params = self.featurizer(total_x)
        z_dim = int(total_z_params.shape[-1] / 2)
        total_z_mu = total_z_params[:, :z_dim]
        total_z_sigma = F.softplus(total_z_params[:, z_dim:])

        z_mu, z_sigma = total_z_mu[: x.shape[0]], total_z_sigma[: x.shape[0]]
        z_mu_target, z_sigma_target = (
            total_z_mu[x.shape[0] :],
            total_z_sigma[x.shape[0] :],
        )

        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample()

        z_dist_target = dist.Independent(
            dist.normal.Normal(z_mu_target, z_sigma_target), 1
        )
        z_target = z_dist_target.rsample()

        preds = torch.softmax(self.classifier(z), 1)
        if self.augment_softmax != 0.0:
            K = 1 - self.augment_softmax * preds.shape[1]
            preds = preds * K + self.augment_softmax
        loss = F.nll_loss(torch.log(preds), y)

        mix_coeff = dist.categorical.Categorical(x.new_ones(x.shape[0]))
        mixture = dist.mixture_same_family.MixtureSameFamily(mix_coeff, z_dist)
        mix_coeff_target = dist.categorical.Categorical(
            x_target.new_ones(x_target.shape[0])
        )
        mixture_target = dist.mixture_same_family.MixtureSameFamily(
            mix_coeff_target, z_dist_target
        )

        obj = loss
        kl = loss.new_zeros([])
        kl_aux = loss.new_zeros([])
        if self.kl_reg != 0.0:
            kl = (
                mixture_target.log_prob(z_target) - mixture.log_prob(z_target)
            ).mean()
            obj = obj + self.kl_reg * kl
        if self.kl_reg_aux != 0.0:
            kl_aux = (mixture.log_prob(z) - mixture_target.log_prob(z)).mean()
            obj = obj + self.kl_reg_aux * kl_aux

        self.optimizer.zero_grad()
        obj.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "kl": kl.item(), "kl_aux": kl_aux.item()}

    def predict(self, x):
        z_params = self.featurizer(x)
        z_dim = int(z_params.shape[-1] / 2)
        z_mu = z_params[:, :z_dim]
        z_sigma = F.softplus(z_params[:, z_dim:])

        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)

        preds = 0.0
        for s in range(self.num_samples):
            z = z_dist.rsample()
            preds += F.softmax(self.classifier(z), 1)
        preds = preds / self.num_samples

        K = 1 - 0.05 * preds.shape[1]
        preds = preds * K + 0.05
        return preds


class KLMeanOnly(Algorithm):
    """
    KL
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(KLMeanOnly, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        self.featurizer = networks.Featurizer(
            input_shape, self.hparams, probabilistic=True
        )
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )

        cls_lr = (
            100 * self.hparams["lr"]
            if hparams["nonlinear_classifier"]
            else self.hparams["lr"]
        )

        self.optimizer = torch.optim.Adam(
            # list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            [
                {
                    "params": self.featurizer.parameters(),
                    "lr": self.hparams["lr"],
                },
                {"params": self.classifier.parameters(), "lr": cls_lr},
            ],
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        self.num_samples = hparams["num_samples"]
        self.kl_reg = hparams["kl_reg"]
        self.kl_reg_aux = hparams["kl_reg_aux"]
        self.augment_softmax = hparams["augment_softmax"]

    def update(self, minibatches, unlabeled=None):
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        x_target = torch.cat(unlabeled)

        total_x = torch.cat([x, x_target])
        total_z_params = self.featurizer(total_x)
        z_dim = int(total_z_params.shape[-1] / 2)
        total_z_mu = total_z_params[:, :z_dim]
        # total_z_sigma = F.softplus(total_z_params[:, z_dim:])
        total_z_sigma = torch.ones_like(total_z_mu)

        z_mu, z_sigma = total_z_mu[: x.shape[0]], total_z_sigma[: x.shape[0]]
        z_mu_target, z_sigma_target = (
            total_z_mu[x.shape[0] :],
            total_z_sigma[x.shape[0] :],
        )

        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample()

        z_dist_target = dist.Independent(
            dist.normal.Normal(z_mu_target, z_sigma_target), 1
        )
        z_target = z_dist_target.rsample()

        preds = torch.softmax(self.classifier(z), 1)
        if self.augment_softmax != 0.0:
            K = 1 - self.augment_softmax * preds.shape[1]
            preds = preds * K + self.augment_softmax
        loss = F.nll_loss(torch.log(preds), y)

        mix_coeff = dist.categorical.Categorical(x.new_ones(x.shape[0]))
        mixture = dist.mixture_same_family.MixtureSameFamily(mix_coeff, z_dist)
        mix_coeff_target = dist.categorical.Categorical(
            x_target.new_ones(x_target.shape[0])
        )
        mixture_target = dist.mixture_same_family.MixtureSameFamily(
            mix_coeff_target, z_dist_target
        )

        obj = loss
        kl = loss.new_zeros([])
        kl_aux = loss.new_zeros([])
        if self.kl_reg != 0.0:
            kl = (
                mixture_target.log_prob(z_target) - mixture.log_prob(z_target)
            ).mean()
            obj = obj + self.kl_reg * kl
        if self.kl_reg_aux != 0.0:
            kl_aux = (mixture.log_prob(z) - mixture_target.log_prob(z)).mean()
            obj = obj + self.kl_reg_aux * kl_aux

        self.optimizer.zero_grad()
        obj.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "kl": kl.item(), "kl_aux": kl_aux.item()}

    def predict(self, x):
        z_params = self.featurizer(x)
        z_dim = int(z_params.shape[-1] / 2)
        z_mu = z_params[:, :z_dim]
        z_sigma = torch.ones_like(z_mu)
        # z_sigma = F.softplus(z_params[:, z_dim:])

        z_dist = dist.Independent(dist.normal.Normal(z_mu, z_sigma), 1)

        preds = 0.0
        for s in range(self.num_samples):
            z = z_dist.rsample()
            preds += F.softmax(self.classifier(z), 1)
        preds = preds / self.num_samples

        K = 1 - 0.05 * preds.shape[1]
        preds = preds * K + 0.05
        return preds


class WD(Algorithm):
    """Wasserstein Distance guided Representation Learning"""

    def __init__(
        self,
        input_shape,
        num_classes,
        num_domains,
        hparams,
        class_balance=False,
    ):
        super(WD, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )

        self.fw = networks.MLP(self.featurizer.n_outputs, 1, self.hparams)

        # Optimizers
        self.wd_opt = torch.optim.Adam(
            self.fw.parameters(),
            lr=self.hparams["lr_wd"],
            weight_decay=self.hparams["weight_decay_wd"],
        )

        self.main_opt = torch.optim.Adam(
            (
                list(self.featurizer.parameters())
                + list(self.classifier.parameters())
            ),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def wd_loss(self, h_s, h_t, for_fw=True):
        batch_size = h_s.shape[0]
        alpha = torch.rand([batch_size, 1]).to(h_s.device)
        h_inter = h_s * alpha + h_t * (1 - alpha)
        h_whole = torch.cat([h_s, h_t, h_inter], 0)
        critic = self.fw(h_whole)

        critic_s = critic[: h_s.shape[0]]
        critic_t = critic[h_s.shape[0] : h_s.shape[0] + h_t.shape[0]]
        wd_loss = critic_s.mean() - critic_t.mean()

        if for_fw is False:
            return wd_loss
        else:
            epsilon = 1e-10  # for stable torch.sqrt
            grad = autograd.grad(critic.sum(), [h_whole], create_graph=True)[0]
            grad_penalty = (
                (torch.sqrt((grad**2).sum(dim=1) + epsilon) - 1) ** 2
            ).mean(dim=0)
            return -wd_loss + self.hparams["grad_penalty"] * grad_penalty

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        features_target = [self.featurizer(xit) for xit in unlabeled]
        total_features = features + features_target
        total_d = len(total_features)

        for _ in range(self.hparams["wd_steps_per_step"]):
            # train fw
            fw_loss = 0.0
            for i in range(total_d):
                for j in range(i + 1, total_d):
                    fw_loss += self.wd_loss(
                        total_features[i], total_features[j], True
                    )
            fw_loss /= total_d * (total_d - 1) / 2
            self.wd_opt.zero_grad()
            fw_loss.backward(retain_graph=True)
            self.wd_opt.step()

        # Train main network
        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
        for i in range(total_d):
            for j in range(i + 1, total_d):
                penalty += self.wd_loss(
                    total_features[i], total_features[j], False
                )

        objective /= nmb
        if nmb > 1:
            penalty /= total_d * (total_d - 1) / 2

        self.main_opt.zero_grad()
        (objective + (self.hparams["lambda_wd"] * penalty)).backward()
        self.main_opt.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(Algorithm):
    """Domain-Adversarial Neural Networks"""

    def __init__(
        self,
        input_shape,
        num_classes,
        num_domains,
        hparams,
        class_balance=False,
    ):
        super(DANN, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        self.register_buffer("update_count", torch.tensor([0]))
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams["nonlinear_classifier"],
        )
        self.discriminator = networks.MLP(
            self.featurizer.n_outputs, num_domains, self.hparams
        )
        self.class_embeddings = nn.Embedding(
            num_classes, self.featurizer.n_outputs
        )

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (
                list(self.discriminator.parameters())
                + list(self.class_embeddings.parameters())
            ),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = torch.optim.Adam(
            (
                list(self.featurizer.parameters())
                + list(self.classifier.parameters())
            ),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device

        self.update_count += 1
        x_each_domain = [x for x, y in minibatches] + unlabeled
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])
        x_target = torch.cat(unlabeled)
        total_x = torch.cat([x, x_target])
        total_z = self.featurizer(total_x)

        z = total_z[: x.shape[0]]

        disc_input = total_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [
                torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
                for i, x in enumerate(x_each_domain)
            ]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(
            disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True
        )[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            preds = self.classifier(z)
            classifier_loss = F.cross_entropy(preds, y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, gaussian
    ):
        super(AbstractMMD, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        #  res = torch.sum((x1[:, None] - x2[None, :]) ** 2, -1)
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        inputs = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])

        source_features = self.featurizer(inputs)
        target_features = self.featurizer(torch.cat(unlabeled))

        objective = F.cross_entropy(self.classifier(source_features), labels)
        penalty = self.mmd(source_features, target_features)

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=False
        )


class AbstractIPMBound(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using the IPM Bound Approach (abstract class)
    """

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, gaussian
    ):
        super(AbstractIPMBound, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "linear"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def weighted_mmd(self, x, w_x, y, w_y):
        if self.kernel_type == "gaussian":
            Wxx = w_x[:, None] * w_x[None, :]
            Wyy = w_y[:, None] * w_y[None, :]
            Wxy = w_x[:, None] * w_y[None, :]

            Kxx = (Wxx * self.gaussian_kernel(x, x)).mean()
            Kyy = (Wyy * self.gaussian_kernel(y, y)).mean()
            Kxy = (Wxy * self.gaussian_kernel(x, y)).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            raise NotImplementedError()

    def iw_unconstrained(
        self,
        source,
        target,
        loss,
        upper,
        l2reg,
        alpha,
        gamma,
        device,
    ):
        Kxx = self.gaussian_kernel(source, source)
        Kxy = self.gaussian_kernel(source, target)

        n_source = len(source)

        C = 2
        l2reg_mat = n_source * l2reg * torch.eye(n_source)
        l2reg_mat = l2reg_mat.to(device)

        A = (C / n_source) * Kxx + l2reg_mat
        b = C * Kxy.mean(dim=-1)

        w = torch.linalg.inv(A) @ b
        w = torch.clamp(w, min=1e-30, max=upper)

        w = w / w.mean()

        return w

    def iwa_unconstrained(
        self,
        source,
        target,
        loss,
        upper,
        l2reg,
        alpha,
        gamma,
        device,
    ):
        Kxx = self.gaussian_kernel(source, source)
        Kxy = self.gaussian_kernel(source, target)

        n_source = len(source)

        A1 = (1 - alpha) * (1 - alpha) / n_source * Kxx
        A2 = n_source * l2reg * torch.eye(n_source).to(device)
        A = A1 + A2

        b1 = (1 - alpha) * Kxy.mean(dim=-1)
        b2 = alpha * (1 - alpha) * Kxx.mean(dim=-1)
        b = b1 - b2

        w = torch.linalg.inv(A) @ b
        w = torch.clamp(w, min=1e-30, max=upper)

        w = w / w.mean()

        return w

    def iwb_unconstrained(
        self,
        source,
        target,
        loss,
        upper,
        l2reg,
        alpha,
        gamma,
        device,
    ):
        Kxx = self.gaussian_kernel(source, source)
        Kxy = self.gaussian_kernel(source, target)

        n_source = len(source)

        A1 = (1 - alpha) * (1 - alpha) / n_source * Kxx
        A2 = n_source * l2reg * torch.eye(n_source).to(device)
        A = A1 + A2

        b1 = (1 - alpha) * Kxy.mean(dim=-1)
        b2 = alpha * (1 - alpha) * Kxx.mean(dim=-1)
        b3 = (loss / gamma) / 2
        b = b1 - b2 - b3

        w = torch.linalg.inv(A) @ b
        w = torch.clamp(w, min=1e-30, max=upper)

        w = w / w.mean()

        return w

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device

        inputs = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])

        source_features = self.featurizer(inputs)
        target_features = self.featurizer(torch.cat(unlabeled))
        losses = F.cross_entropy(
            self.classifier(source_features), labels, reduction="none"
        )

        with torch.no_grad():
            # _s_w = self.iw_unconstrained(  # Works decent
            _s_w = self.iwa_unconstrained(
                source_features,
                target_features,
                losses,
                self.hparams["upper"],
                self.hparams["iw_l2_reg"],
                self.hparams["iw_mixture"],
                self.hparams["mmd_gamma"],
                device,
            )

        t_w = torch.ones(target_features.shape[0]).to(device)

        s_w = (
            self.hparams["iw_mixture"]
            * torch.ones(source_features.shape[0]).to(device)
            + (1 - self.hparams["iw_mixture"]) * _s_w
        )

        objective = (s_w * losses).mean()
        penalty = self.weighted_mmd(source_features, s_w, target_features, t_w)

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "loss": objective.item(),
            "penalty": penalty,
            "min_weight": s_w.min().item(),
            "max_weight": s_w.max().item(),
        }


class MMDBound(AbstractIPMBound):
    """
    IPMBound using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMDBound, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


###############################################################################
###############################################################################
################### Graveyard #################################################
###############################################################################
###############################################################################


class AbstractPMMD(PERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, gaussian
    ):
        super(AbstractPMMD, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"

    def my_cdist(self, x1, x2):
        #  res = torch.sum((x1[:, None] - x2[None, :]) ** 2, -1)
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            raise NotImplementedError()

    def batched_kernel_mean(self, x, y):
        # x = B x 64 x F
        # Want x, x => B x B x F x F => mean last two => B x B => mean
        total = 0
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                total += self.gaussian_kernel(x[i], y[j]).mean()

        return total / (x.shape[0] * y.shape[0])

    def multi_avg_mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.batched_kernel_mean(x, x)
            Kyy = self.batched_kernel_mean(y, y)
            Kxy = self.batched_kernel_mean(x, y)
            return Kxx + Kyy - 2 * Kxy
        else:
            raise NotImplementedError()

    def empirical_prob_mmd(self, x, p_x, y, p_y):
        if self.kernel_type == "gaussian":
            Pxx = p_x[:, None] * p_x[None, :]
            Pyy = p_y[:, None] * p_y[None, :]
            Pxy = p_x[:, None] * p_y[None, :]

            Kxx = (Pxx * self.gaussian_kernel(x, x)).sum()
            Kyy = (Pyy * self.gaussian_kernel(y, y)).sum()
            Kxy = (Pxy * self.gaussian_kernel(x, y)).sum()
            return Kxx + Kyy - 2 * Kxy
        else:
            raise NotImplementedError()

    def update(self, minibatches, unlabeled=None):
        source_inputs = torch.cat([x for x, y in minibatches])
        source_labels = torch.cat([y for x, y in minibatches])

        target_inputs = torch.cat(unlabeled)

        _source_features = self.featurizer(source_inputs)
        _target_features = self.featurizer(target_inputs)

        z_dim = int(_source_features.shape[-1] / 2)

        source_z_mu = _source_features[:, :z_dim]
        source_z_sigma = F.softplus(_source_features[:, z_dim:])

        target_z_mu = _target_features[:, :z_dim]
        target_z_sigma = F.softplus(_target_features[:, z_dim:])

        source_z_dist = dist.Independent(
            dist.normal.Normal(source_z_mu, source_z_sigma), 1
        )
        target_z_dist = dist.Independent(
            dist.normal.Normal(target_z_mu, target_z_sigma), 1
        )

        source_features = source_z_dist.rsample()
        # target_features = target_z_dist.rsample()

        objective = F.cross_entropy(
            self.classifier(source_features), source_labels
        )

        resampled_source_features = source_z_dist.rsample((32,))
        resampled_target_features = target_z_dist.rsample((32,))

        resampled_source_features = torch.transpose(
            resampled_source_features, 0, 1
        )
        resampled_target_features = torch.transpose(
            resampled_target_features, 0, 1
        )

        penalty = self.multi_avg_mmd(
            resampled_source_features,
            resampled_target_features,
        )

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class PMMD(AbstractPMMD):
    """
    PMMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PMMD, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )


class PERMIW(PERM):
    """
    Empirical Risk Minimization (ERM) with probabilistic representation network and importance weighting
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PERMIW, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

    def update(self, minibatches, unlabeled=None):
        source_inputs = torch.cat([x for x, y in minibatches])
        source_labels = torch.cat([y for x, y in minibatches])

        target_inputs = torch.cat(unlabeled)

        _source_features = self.featurizer(source_inputs)
        _target_features = self.featurizer(target_inputs)

        z_dim = int(_source_features.shape[-1] / 2)

        source_z_mu = _source_features[:, :z_dim]
        source_z_sigma = F.softplus(_source_features[:, z_dim:])

        target_z_mu = _target_features[:, :z_dim]
        target_z_sigma = F.softplus(_target_features[:, z_dim:])

        source_z_dist = dist.Independent(
            dist.normal.Normal(source_z_mu, source_z_sigma), 1
        )
        target_z_dist = dist.Independent(
            dist.normal.Normal(target_z_mu, target_z_sigma), 1
        )

        source_mix_coeff = dist.categorical.Categorical(
            source_inputs.new_ones(source_inputs.shape[0])
        )
        target_mix_coeff = dist.categorical.Categorical(
            target_inputs.new_ones(target_inputs.shape[0])
        )

        source_mixture = dist.mixture_same_family.MixtureSameFamily(
            source_mix_coeff, source_z_dist
        )
        target_mixture = dist.mixture_same_family.MixtureSameFamily(
            target_mix_coeff, target_z_dist
        )

        source_latent = source_z_dist.rsample()
        with torch.no_grad():
            w = torch.exp(
                target_mixture.log_prob(source_latent)
                - source_mixture.log_prob(source_latent)
            )
            w = w + 1e-30  # Numeric stability
            w = w / w.mean()

            # Maybe do sometype of annealing process on alpha
            # alpha = 0.01
            alpha = 0.0
            w = alpha + (1 - alpha) * w

        losses = F.cross_entropy(
            self.classifier(source_latent), source_labels, reduction="none"
        )

        loss = (w * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "min_weight": w.min().item(),
            "max_weight": w.max().item(),
            #            "mean_source_z_mu": source_z_mu.mean().item(),
            #            "mean_target_z_mu": target_z_mu.mean().item(),
            #            "mean_source_z_sigma": source_z_sigma.mean().item(),
            #            "mean_target_z_sigma": target_z_sigma.mean().item(),
        }


class AbstractPIPMBound(PERM):
    """
    Perform PERM while matching the pair-wise domain feature distributions
    using the IPM Bound Approach (abstract class)
    """

    def __init__(
        self, input_shape, num_classes, num_domains, hparams, gaussian
    ):
        super(AbstractPIPMBound, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "linear"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def weighted_mmd(self, x, w_x, y, w_y):
        if self.kernel_type == "gaussian":
            Wxx = w_x[:, None] * w_x[None, :]
            Wyy = w_y[:, None] * w_y[None, :]
            Wxy = w_x[:, None] * w_y[None, :]

            Kxx = (Wxx * self.gaussian_kernel(x, x)).mean()
            Kyy = (Wyy * self.gaussian_kernel(y, y)).mean()
            Kxy = (Wxy * self.gaussian_kernel(x, y)).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            raise NotImplementedError()

    def iw_unconstrained(
        self,
        source,
        target,
        loss,
        upper,
        l2reg,
        device,
    ):
        Kxx = self.gaussian_kernel(source, source)
        Kxy = self.gaussian_kernel(source, target)

        n_source = len(source)

        C = 2
        l2reg_mat = n_source * l2reg * torch.eye(n_source)
        l2reg_mat = l2reg_mat.to(device)

        A = (C / n_source) * Kxx + l2reg_mat
        b = C * Kxy.mean(axis=-1)

        w = torch.linalg.inv(A) @ b
        w = torch.clamp(w, min=1e-30, max=upper)

        w = w / w.mean()

        return w

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device

        source_inputs = torch.cat([x for x, y in minibatches])
        source_labels = torch.cat([y for x, y in minibatches])

        target_inputs = torch.cat(unlabeled)

        _source_features = self.featurizer(source_inputs)
        _target_features = self.featurizer(target_inputs)

        z_dim = int(_source_features.shape[-1] / 2)

        source_z_mu = _source_features[:, :z_dim]
        source_z_sigma = F.softplus(_source_features[:, z_dim:])

        target_z_mu = _target_features[:, :z_dim]
        target_z_sigma = F.softplus(_target_features[:, z_dim:])

        source_z_dist = dist.Independent(
            dist.normal.Normal(source_z_mu, source_z_sigma), 1
        )
        target_z_dist = dist.Independent(
            dist.normal.Normal(target_z_mu, target_z_sigma), 1
        )

        source_mix_coeff = dist.categorical.Categorical(
            source_inputs.new_ones(source_inputs.shape[0])
        )
        target_mix_coeff = dist.categorical.Categorical(
            target_inputs.new_ones(target_inputs.shape[0])
        )

        source_mixture = dist.mixture_same_family.MixtureSameFamily(
            source_mix_coeff, source_z_dist
        )
        target_mixture = dist.mixture_same_family.MixtureSameFamily(
            target_mix_coeff, target_z_dist
        )

        source_latents = source_z_dist.rsample()
        target_latents = target_z_dist.rsample()

        with torch.no_grad():
            s_w = torch.exp(
                target_mixture.log_prob(source_latents)
                - source_mixture.log_prob(source_latents)
            )
            s_w = s_w + 1e-30  # Numeric stability
            s_w = s_w / s_w.mean()

            # Maybe do sometype of annealing process on alpha
            # alpha = 0.01
            alpha = self.hparams["iw_mixture"]
            s_w = alpha + (1 - alpha) * s_w

        t_w = torch.ones(target_inputs.shape[0]).to(device)

        losses = F.cross_entropy(
            self.classifier(source_latents), source_labels, reduction="none"
        )

        objective = (s_w * losses).mean()
        penalty = self.weighted_mmd(source_latents, s_w, target_latents, t_w)

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {
            "loss": objective.item(),
            "penalty": penalty,
            "min_weight": s_w.min().item(),
            "max_weight": s_w.max().item(),
        }


class PMMDBound(AbstractPIPMBound):
    """
    IPMBound using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PMMDBound, self).__init__(
            input_shape, num_classes, num_domains, hparams, gaussian=True
        )
