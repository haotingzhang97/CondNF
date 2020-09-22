from colorization import *
from skimage.metrics import structural_similarity as ssim
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



def np_L1loss(x, y):
    return np.mean(np.absolute(np.array(x)-np.array(y)))

def ssim_batch(x, y):
    ssim_val = 0
    n = np.shape(x)[0]
    for i in range(n):
        xx = np.swapaxes(np.swapaxes(x[i, :, :, :], 0, 1), 1, 2)
        yy = np.swapaxes(np.swapaxes(y[i, :, :, :], 0, 1), 1, 2)
        ssim_val += ssim(xx, yy, multichannel=True)
    ssim_val /= n
    return ssim_val

def ged_mnist(data_x, target_x, p, model, opt, n_samples=100, n_each_digit=1, loss='L1'):
    # Generalised Energy Distance:
    # D^2_{GED}(P_{gt}, P_{out}) = 2E[d(S,Y)] - E[d(S,S')] - E[d(Y,Y')]
    # S,S' -- independent samples from the predicted distribution P_{out}
    # Y,Y' -- independent samples from the ground truth distribution P_{gt}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # compute the three ground truth modes for all items in the test set (data_x to data_gt) [size: n,1,p,p]
    dim_x = np.shape(data_x)[2]
    ged = np.zeros((10))   # compute ged separately for the 10 digits
    div0 = np.zeros((10))
    for digit in range(10):
        print(digit)
        ind = np.where(target_x == digit)[0]
        i = ind[np.random.randint(0, len(ind), n_each_digit)]
        data_gt_red = colorize_red(data_x[i,:,:,:], d=1, output_type='np')
        data_gt_green = colorize_green(data_x[i,:,:,:], d=1, output_type='np')
        data_gt_blue = colorize_blue(data_x[i,:,:,:], d=1, output_type='np')
        if loss == 'L1':
            dYY = 2 * p[digit, 0] * p[digit, 1] * np_L1loss(data_gt_red, data_gt_green) + \
            2 * p[digit, 0] * p[digit, 2] * np_L1loss(data_gt_red, data_gt_blue) + \
            2 * p[digit, 1] * p[digit, 2] * np_L1loss(data_gt_green, data_gt_blue)
        if loss == 'mssim':
            dYY = 2 * p[digit, 0] * p[digit, 1] * (1-ssim_batch(data_gt_red, data_gt_green)) + \
            2 * p[digit, 0] * p[digit, 2] * (1-ssim_batch(data_gt_red, data_gt_blue)) + \
            2 * p[digit, 1] * p[digit, 2] * (1-ssim_batch(data_gt_green, data_gt_blue))
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
            if loss == 'L1':
                dSY = p[digit, 0] * np_L1loss(data_gt_red, output) + p[digit, 1] * np_L1loss(data_gt_green, output) + \
                    p[digit, 2] * np_L1loss(data_gt_blue, output)
            if loss == 'mssim':
                dSY = p[digit, 0] * (1-ssim_batch(data_gt_red, output)) + \
                    p[digit, 1] * (1-ssim_batch(data_gt_green, output)) + \
                    p[digit, 2] * (1-ssim_batch(data_gt_blue, output))
        if opt.model_name == 'pix2pix' or opt.model_name == 'cglow':
            output_mat = np.zeros((n_samples, n_each_digit, 3, dim_x, dim_x))
            for k in range(n_samples):
                input = data_x[i, :, :, :].to(device).float()
                model.isTrain = False
                if opt.model_name == 'pix2pix':
                    model.set_input(input)
                    if device == 'cuda':
                        output = model.forward().detach().cpu().numpy()
                    else:
                        output = model.forward().detach().numpy()
                else:
                    output, _ = model(input, reverse=True)
                    if device == 'cuda':
                        output = output.detach().cpu().numpy()
                    else:
                        output = output.detach().numpy()
                output_mat[k,:,:,:,:] = output
                if loss == 'L1':
                    dSY += p[digit, 0] * np_L1loss(data_gt_red, output) + p[digit, 1] * \
                          np_L1loss(data_gt_green, output) + p[digit, 2] * np_L1loss(data_gt_blue, output)
                if loss == 'mssim':
                    dSY += p[digit, 0] * (1-ssim_batch(data_gt_red, output)) + \
                          p[digit, 1] * (1-ssim_batch(data_gt_green, output)) + \
                          p[digit, 2] * (1-ssim_batch(data_gt_blue, output))
            dSY /= n_samples
            for k1 in range(n_samples):
                for k2 in range(n_samples):
                    if loss == 'L1':
                        dSS += np.mean(np.absolute(output_mat[k1, :, :, :, :] - output_mat[k2, :, :, :, :]))
                        #dSS += np_L1loss(output_mat[k1, :, :, :, :], output[k2, :, :, :, :])
                    if loss == 'mssim':
                        dSS += 1 - ssim_batch(output_mat[k1, :, :, :, :], output_mat[k2, :, :, :, :])
            dSS /= (n_samples * n_samples)

        if opt.model_name == 'MSGAN':
            output_mat = np.zeros((n_samples, n_each_digit, 3, dim_x, dim_x))
            for kk in range(len(i)):
                ij = i[kk]
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
                    output_mat[k,kk,:,:,:] = fake_B
                    data_gt_red_k = np.expand_dims(data_gt_red[k, :, :, :], 0)
                    data_gt_green_k = np.expand_dims(data_gt_green[k, :, :, :], 0)
                    data_gt_blue_k = np.expand_dims(data_gt_blue[k, :, :, :], 0)
                    if loss == 'L1':
                        dSY += p[digit, 0] * np_L1loss(data_gt_red_k, fake_B) + p[digit, 1] * np_L1loss(data_gt_green_k, fake_B) + \
                               p[digit, 2] * np_L1loss(data_gt_blue_k, fake_B)
                    if loss == 'mssim':
                        dSY += p[digit, 0] * (1-ssim_batch(data_gt_red_k, fake_B)) + p[digit, 1] * \
                               (1-ssim_batch(data_gt_green_k, fake_B)) + p[digit, 2] * \
                               (1-ssim_batch(data_gt_blue_k, fake_B))
            dSY /= (n_samples*n_each_digit)
            for k1 in range(n_samples):
                for k2 in range(n_samples):
                    if loss == 'L1':
                        dSS += np.mean(np.absolute(output_mat[k1, :, :, :, :] - output_mat[k2, :, :, :, :]))
                        #dSS += np_L1loss(output_mat[i, :, :, :, :], output[j, :, :, :, :])
                    if loss == 'mssim':
                        dSS += 1 - ssim_batch(output_mat[k1, :, :, :, :], output_mat[k2, :, :, :, :])
            dSS /= (n_samples * n_samples)
        div0[digit] = dSS
        ged[digit] = 2 * dSY - dSS - dYY
        print(dSY, dSS, dYY)
    return ged, div0