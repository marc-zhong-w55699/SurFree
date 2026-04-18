import copy
import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import atleast_kdim


def clip_image_values(x, lb, ub):
    return torch.clamp(x, lb, ub)


def get_init_with_noise(model, X, y):
    init = X.clone()
    p = model(X).argmax(1)

    while any(p == y):
        init = torch.where(
            atleast_kdim(p == y, len(X.shape)), 
            (X + 0.5*torch.randn_like(X)).clip(0, 1), 
            init)
        p = model(init).argmax(1)
    return init


def valid_bounds(img, delta=255):
    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.int32)

    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)

    # Compute the bounds
    lb = im - delta
    ub = im + delta

    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))

    # Change types to uint8
    lb = lb.astype(np.uint8)
    ub = ub.astype(np.uint8)

    return lb, ub

class init_attack:
    def __init__(self, model, src_img, mean, std, lb, ub, 
                 tar_img = None, dim_reduc_factor=4, attack_method = 'surfree',
                 iteration=93, initial_query=30, tol=0.0001, sigma=0.0002, 
                 verbose_control='Yes'):
        self.model = model
        self.src_img = src_img
        self.src_lbl = torch.argmax(self.model.forward(Variable(self.src_img, requires_grad=True)).data).item()
        self.tar_img = tar_img
        if tar_img != None:
            self.tar_lbl = torch.argmax(self.model.forward(Variable(self.tar_img, requires_grad=True)).data).item()
        self.dim_reduc_factor = dim_reduc_factor
        self.iteration =iteration
        self.N0 = initial_query
        self.mean = mean
        self.std = std
        self.lb = lb
        self.ub = ub
        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = 40
        self.verbose_control = verbose_control
        self.attack_method = attack_method

        # print(f'Source imge lbl: {self.src_lbl}     Targeted image lbl: {self.tar_lbl}')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0

    def is_adversarial(self, image):
        predict_label = torch.argmax(self.model.forward(Variable(image, requires_grad=True)).data).item()
        self.all_queries += 1
        if self.tar_img is None:
            is_adv = predict_label != self.src_lbl
        else:
            is_adv = predict_label == self.tar_lbl
        return 1 if is_adv else -1

    def find_random_adversarial(self, image, step=3.0, eps_max=15, n=60):
        num_calls = 0
        perturbed = image
        candidate = image
        max_calls = 50
        for _ in range(n):
            u = torch.randn(image.shape).to(self.device)
            u = u / torch.norm(u)

            eps = step
            candidate = clip_image_values(candidate + eps * u, self.lb, self.ub).to(self.device)
            is_adv = self.is_adversarial(candidate)
            num_calls += 1

            while is_adv == -1 and eps <= eps_max:
                eps += step
                candidate = clip_image_values(candidate + eps * u, self.lb, self.ub).to(self.device)
                is_adv = self.is_adversarial(candidate)
                num_calls += 1

            if is_adv == 1:
                perturbed = candidate
                x_b, bin_calls = self.bin_search(image, perturbed, max_calls)
                num_calls += bin_calls
                return x_b, num_calls

        print("Warning: find_random_adversarial failed to find an adversarial direction after {} trials, returning original image.".format(n))
        return perturbed, num_calls

    def bin_search(self, x_0, x_random, max_calls=100):
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1
            if self.is_adversarial(mid) == 1:
                adv = mid
            else:
                cln = mid
            if torch.norm(adv - cln).cpu().numpy() < self.tol or num_calls >= max_calls:
                break
        return adv, num_calls
