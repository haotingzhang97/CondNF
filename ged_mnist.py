from colorization import *


def np_L1loss(x,y):
    return np.mean(np.absolute(np.array(x)-np.array(y)))


def ged_mnist(data_x, target_x, p, model, opt, n_samples=100, n_each_digit=10):
    # Generalised Energy Distance:
    # D^2_{GED}(P_{gt}, P_{out}) = 2E[d(S,Y)] - E[d(S,S')] - E[d(Y,Y')]
    # S,S' -- independent samples from the predicted distribution P_{out}
    # Y,Y' -- independent samples from the ground truth distribution P_{gt}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # compute the three ground truth modes for all items in the test set (data_x to data_gt) [size: n,1,p,p]
    dim_x = np.shape(data_x)[2]
    ged = np.zeros((10))   # compute ged separately for the 10 digits
    for digit in range(10):
        ind = np.where(target_x == digit)[0]
        i = ind[np.random.randint(0, len(ind), n_each_digit)]
        data_gt_red = colorize_red(data_x[i,:,:,:], d=1, output_type='np')
        data_gt_green = colorize_green(data_x[i,:,:,:], d=1, output_type='np')
        data_gt_blue = colorize_blue(data_x[i,:,:,:], d=1, output_type='np')
        dYY = 2 * p[digit, 0] * p[digit, 1] * np_L1loss(data_gt_red, data_gt_green) + \
        2 * p[digit, 0] * p[digit, 2] * np_L1loss(data_gt_red, data_gt_blue) + \
        2 * p[digit, 1] * p[digit, 2] * np_L1loss(data_gt_green, data_gt_blue)
        dSY = 0
        dSS = 0
        if opt.model_name == 'unet':
            input = data_x[i, :, :, :].to(device).float()
            model.isTrain = False
            model.set_input(input)
            if device == 'cuda':
                output = model.forward().detach().cpu().numpy()
            else:
                output = model.forward().detach().numpy()
            dSY = p[digit, 0] * np_L1loss(data_gt_red, output) + p[digit, 1] * np_L1loss(data_gt_green, output) + \
                p[digit, 2] * np_L1loss(data_gt_blue, output)
        if opt.model_name == 'pix2pix':
            output_mat = np.zeros((n_samples, n_each_digit, 3, dim_x, dim_x))
            for k in range(n_samples):
                input = data_x[i, :, :, :].to(device).float()
                model.isTrain = False
                model.set_input(input)
                if device == 'cuda':
                    output = model.forward().detach().cpu().numpy()
                else:
                    output = model.forward().detach().numpy()
                output_mat[k,:,:,:,:] = output
                dSY += p[digit, 0] * np_L1loss(data_gt_red, output) + p[digit, 1] * np_L1loss(data_gt_green, output) + \
                      p[digit, 2] * np_L1loss(data_gt_blue, output)
            dSY /= n_samples
            for k1 in range(n_samples):
                for k2 in range(n_samples):
                    dSS += np.mean(np.absolute(output_mat[k1, :, :, :, :] - output_mat[k2, :, :, :, :]))
                    #dSS += np_L1loss(output_mat[k1, :, :, :, :], output[k2, :, :, :, :])
            dSS /= (n_samples * n_samples)

        if opt.model_name == 'MSGAN':
            output_mat = np.zeros((n_samples, n_each_digit, 3, dim_x, dim_x))
            for ij in i:
                input = torch.unsqueeze(data_x[ij, :, :, :], axis=0).to(device).float()
                model.isTrain = False
                model.set_input(input)
                nz = 8
                z_samples = model.get_z_random(n_samples, nz)
                for k in range(n_samples):
                    _, fake_B, _ = model.test(z_samples[[k]])
                    if device == 'cuda':
                        fake_B = fake_B.detach().cpu().numpy()
                    else:
                        fake_B = fake_B.detach().numpy()
                    output_mat[k,:,:,:,:] = fake_B
                    dSY += p[digit, 0] * np_L1loss(data_gt_red, fake_B) + p[digit, 1] * np_L1loss(data_gt_green, fake_B) + \
                           p[digit, 2] * np_L1loss(data_gt_blue, fake_B)
            dSY /= (n_samples*n_each_digit)
            for k1 in range(n_samples):
                for k2 in range(n_samples):
                    dSS += np.mean(np.absolute(output_mat[k1, :, :, :, :] - output_mat[k2, :, :, :, :]))
                    #dSS += np_L1loss(output_mat[i, :, :, :, :], output[j, :, :, :, :])
            dSS /= (n_samples * n_samples)
        ged[digit] = np.sqrt(2 * dSY - dSS - dYY)
        print(dSY, dSS, dYY)
    return ged