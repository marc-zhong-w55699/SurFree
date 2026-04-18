import torch
from utils.utils import atleast_kdim


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
def bin_search(self, x_0, x_random,max_calls=100):  
    num_calls = 0
    adv = x_random
    cln = x_0      
    while True:         
        mid = (cln + adv) / 2.0
        num_calls += 1           
        if self.is_adversarial(mid)==1:
            adv = mid
        else:
            cln = mid   
        if torch.norm(adv-cln).cpu().numpy()<self.tol or num_calls>=max_calls:
            break       
    return adv, num_calls 
    
def find_random_adversarial(self, image, step=3.0, eps_max=15, n=60):
    num_calls = 0
    perturbed = image
    candidate = image
    max_calls=50
    for _ in range(n):
        # Sample a unit direction u ~ N(0, I_d)
        u = torch.randn(image.shape).to(self.device)
        u = u / torch.norm(u)

        # Walk along u until adversarial or distance exceeds eps_max
        #eps = 0.01
        eps = step
        candidate = clip_image_values(candidate + eps * u, self.lb, self.ub).to(self.device)
        is_adv = self.is_adversarial(candidate)
        num_calls += 1

        while is_adv == -1 and eps <= eps_max:
            eps += step
            candidate = clip_image_values(candidate + eps * u, self.lb, self.ub).to(self.device)
            is_adv = self.is_adversarial(candidate)
            num_calls += 1

        # If adversarial point found, binary-search back to the boundary
        if is_adv == 1:
            perturbed = candidate
            x_b, bin_calls = self.bin_search(image, perturbed,max_calls)
            num_calls += bin_calls
            return x_b, num_calls

    print("Warning: find_random_adversarial failed to find an adversarial direction after {} trials, returning original image.".format(n))
    return perturbed, num_calls