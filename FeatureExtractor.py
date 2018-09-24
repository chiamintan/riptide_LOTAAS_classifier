import numpy as np
import sys

class FeatureExtractor:

    def __init__(self):

        self.feature = []

    def scale(self, data):

        #scale the values of the input array to be between 0 and 255 (currently not used)

        data_scaled = (data-np.min(data))*255/(np.max(data)-np.min(data))
        return data_scaled

    def rebin(self, data, nbin):

        #rebin the data into number of bins desired

        if data.ndim == 2:
            scrunchfactor = len(data[0])/nbin
            data_scrunched = []
            for j in range(len(data)):
                data_s = ([sum(data[j][i:i+scrunchfactor])/scrunchfactor for i in xrange(0, len(data[j]), scrunchfactor)])
                data_scrunched.append(data_s)
        elif data.ndim == 1:
            scrunchfactor = len(data) / nbin
            data_scrunched = [sum(data[i:i+scrunchfactor])/scrunchfactor for i in xrange(0, len(data), scrunchfactor)]
        else:
            print "There shouldn't be data with other dimensions"
            sys.exit()

        data_scrunched = np.asarray(data_scrunched)
        return data_scrunched

    def gated(self, subints, profile):

        #gating the profile and subints to 25% of phase around maximum

        shift_bins = profile.size // 2 - profile.argmax()

        profile_shift = np.roll(profile, shift_bins)
        subints_shift = np.roll(subints, shift_bins, axis=1)

        binstart = int(0.375*len(profile_shift))
        binend = int(0.625*len(profile_shift))

        subints_gated = subints_shift[:,binstart:binend]
        profile_gated = profile_shift[binstart:binend]

        return subints_gated, profile_gated

    def skewness(self, obs):

        num = np.sum((obs - np.mean(obs)) ** 3) / len(obs)
        denom = (np.sqrt(np.sum((obs - np.mean(obs)) ** 2) / len(obs))) ** 3
        return (num / denom)

    def excess_kurtosis(self, obs):

        num = np.sum((obs - np.mean(obs)) ** 4) / len(obs)
        denom = ((np.sum((obs - np.mean(obs)) ** 2)) / len(obs)) ** 2
        return (num / denom) - 3

    def features_shape(self, trials, SNRs):

        #Calculate the features from curves based on features 17-20 of Tan et al 2018

        norm_trials = (trials - trials[0]) / (trials[-1] - trials[0])

        shapemn = sum([x * y for x, y in zip(SNRs, norm_trials)]) / sum(SNRs)
        trialsminusmn = norm_trials - shapemn
        shapevr = sum([x * y for x, y in zip(SNRs, trialsminusmn ** 2)]) / sum(SNRs)
        shapesd = np.sqrt(shapevr)
        shapeskw = abs((sum([x * y for x, y in zip(SNRs, trialsminusmn ** 3)]) / sum(SNRs)) / shapevr ** 1.5)
        shapekurt = ((sum([x * y for x, y in zip(SNRs, trialsminusmn ** 4)]) / sum(SNRs)) / shapevr ** 2) - 3

        return shapemn, shapesd, shapeskw, shapekurt

    def subint_correlation(self, subints, profile):

        #calculate the correlation coefficients of each subints against the profile

        self.correlation = []

        new_subints = [s for s in subints if np.std(s) >= 5e-7]

        if 48 <= len(new_subints) <= 80:
            scrunchfactor = 2
        elif 81 <= len(new_subints) <= 119:
            scrunchfactor = 3
        elif len(new_subints) >= 120:
            scrunchfactor = 4
        else:
            scrunchfactor = 1

        new_subints_scrunched = [sum(new_subints[i:i+scrunchfactor]) for i in xrange(0, len(new_subints), scrunchfactor)]

        for i in range(len(new_subints_scrunched)):
            self.correlation.append(np.corrcoef(new_subints_scrunched[i],profile)[0,1])

        return self.correlation

    def getfeatures(self, candidate, feature_type):

        if feature_type == 1:

            self.feature.append(np.mean(candidate.subints.normalised_profile))
            self.feature.append(np.std(candidate.subints.normalised_profile))
            self.feature.append(self.skewness(candidate.subints.normalised_profile))
            self.feature.append(self.excess_kurtosis(candidate.subints.normalised_profile))

            self.feature.append(np.mean(candidate.dm_curve.snr))
            self.feature.append(np.std(candidate.dm_curve.snr))
            self.feature.append(self.skewness(candidate.dm_curve.snr))
            self.feature.append(self.excess_kurtosis(candidate.dm_curve.snr))

            self.feature.append(np.mean(candidate.period_curve.snr))
            self.feature.append(np.std(candidate.period_curve.snr))
            self.feature.append(self.skewness(candidate.period_curve.snr))
            self.feature.append(self.excess_kurtosis(candidate.period_curve.snr))

            correlation_coef = self.subint_correlation(candidate.subints.data,candidate.subints.normalised_profile)

            self.feature.append(np.mean(correlation_coef))
            self.feature.append(np.std(correlation_coef))
            self.feature.append(self.skewness(correlation_coef))
            self.feature.append(self.excess_kurtosis(correlation_coef))

            DMshapemn, DMshapesd, DMshapeskw, DMshapekurt = self.features_shape(candidate.dm_curve.trials, candidate.dm_curve.snr)

            self.feature.append(DMshapemn)
            self.feature.append(DMshapesd)
            self.feature.append(DMshapeskw)
            self.feature.append(DMshapekurt)

            return self.feature

        if feature_type == 2:

            subints_gated, profile_gated = self.gated(candidate.subints.data,candidate.subints.normalised_profile)

            self.feature.append(np.mean(profile_gated))
            self.feature.append(np.std(profile_gated))
            self.feature.append(self.skewness(profile_gated))
            self.feature.append(self.excess_kurtosis(profile_gated))

            self.feature.append(np.mean(candidate.dm_curve.snr))
            self.feature.append(np.std(candidate.dm_curve.snr))
            self.feature.append(self.skewness(candidate.dm_curve.snr))
            self.feature.append(self.excess_kurtosis(candidate.dm_curve.snr))

            self.feature.append(np.mean(candidate.period_curve.snr))
            self.feature.append(np.std(candidate.period_curve.snr))
            self.feature.append(self.skewness(candidate.period_curve.snr))
            self.feature.append(self.excess_kurtosis(candidate.period_curve.snr))

            correlation_coef = self.subint_correlation(subints_gated,profile_gated)

            self.feature.append(np.mean(correlation_coef))
            self.feature.append(np.std(correlation_coef))
            self.feature.append(self.skewness(correlation_coef))
            self.feature.append(self.excess_kurtosis(correlation_coef))

            DMshapemn, DMshapesd, DMshapeskw, DMshapekurt = self.features_shape(candidate.dm_curve.trials, candidate.dm_curve.snr)

            self.feature.append(DMshapemn)
            self.feature.append(DMshapesd)
            self.feature.append(DMshapeskw)
            self.feature.append(DMshapekurt)

            return self.feature

        if feature_type == 3:

            profile_rebin = self.rebin(candidate.subints.normalised_profile, 256)
            subints_rebin = self.rebin(candidate.subints.data, 256)

            self.feature.append(np.mean(profile_rebin))
            self.feature.append(np.std(profile_rebin))
            self.feature.append(self.skewness(profile_rebin))
            self.feature.append(self.excess_kurtosis(profile_rebin))

            self.feature.append(np.mean(candidate.dm_curve.snr))
            self.feature.append(np.std(candidate.dm_curve.snr))
            self.feature.append(self.skewness(candidate.dm_curve.snr))
            self.feature.append(self.excess_kurtosis(candidate.dm_curve.snr))

            self.feature.append(np.mean(candidate.period_curve.snr))
            self.feature.append(np.std(candidate.period_curve.snr))
            self.feature.append(self.skewness(candidate.period_curve.snr))
            self.feature.append(self.excess_kurtosis(candidate.period_curve.snr))

            correlation_coef = self.subint_correlation(subints_rebin, profile_rebin)

            self.feature.append(np.mean(correlation_coef))
            self.feature.append(np.std(correlation_coef))
            self.feature.append(self.skewness(correlation_coef))
            self.feature.append(self.excess_kurtosis(correlation_coef))

            DMshapemn, DMshapesd, DMshapeskw, DMshapekurt = self.features_shape(candidate.dm_curve.trials,candidate.dm_curve.snr)

            self.feature.append(DMshapemn)
            self.feature.append(DMshapesd)
            self.feature.append(DMshapeskw)
            self.feature.append(DMshapekurt)

            return self.feature