import random
import time
import numpy as np

def get_synth_ai_human_regions(dataset, metadata_dimensions = 1, GOOD_REGION_ACC = 0.99, BAD_REGION_ACC = 0.2, OTHER_REGION_ACC = 0.6, N_CLAUSES = 2, CLAUSE_LENGTH = 1, MIN_REGION_SIZE = 0.01, MAX_REGION_SIZE = 0.2):
    '''
    args:
        dataset: Dataset object
        metadata_dimensions: number of metadata dimensions
        GOOD_REGION_ACC: accuracy of AI in good regions
        BAD_REGION_ACC: accuracy of AI in bad regions
        OTHER_REGION_ACC: accuracy of AI in other regions
        N_CLAUSES: number of clauses in each region
        CLAUSE_LENGTH: number of metadata dimensions in each clause
        MIN_REGION_SIZE: minimum size of a region
        MAX_REGION_SIZE: maximum size of a region
    returns:
        ai_preds: predictions of AI
        hum_preds: predictions of human
        true_regions: true regions of each point
        ai_regions_names: names of AI regions
        hum_regions_names: names of human regions
    '''
    def get_synth_ai_human_regions_helper():
        start_time = time.time()


        ai_preds = np.zeros(len(dataset.data_y))
        ai_regions_names = []
        ai_regions = []
        ai_regions_raw = []
        hum_preds = np.zeros(len(dataset.data_y))
        hum_regions = []
        hum_regions_names = []
        hum_regions_raw = []
        
        # get AI regions
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > 60:
                return None
            # sample a metadata point
            metadata_point = dataset.metadata[np.random.randint(len(dataset.metadata))]
            # sample CLAUSE_LENGTH metadata dimensions
            clause = np.random.choice(metadata_dimensions, CLAUSE_LENGTH, replace=False)
            # sort clause
            clause = np.sort(clause)
            potential_region = [metadata_point[i] for i in clause]
            if potential_region[0] == "miscellaneous":
                continue
            # count number of points in region
            region_size = 0
            for i in range(len(dataset.metadata)):
                if all(dataset.metadata[i][j] == potential_region[k] for k,j in enumerate(clause)):
                    region_size += 1
            # check if region is too small or too big
            if region_size < MIN_REGION_SIZE*len(dataset.metadata) or region_size > MAX_REGION_SIZE*len(dataset.metadata):
                continue
            # check if region already exists
            ai_good = len(ai_regions) < N_CLAUSES/2
            # included in other region
            included_in_other_region = False
            for other_region in ai_regions_raw:
                # check crosswise inclusion
                for i in range(len(clause)):
                    for j in range(len(clause)):
                        if potential_region[i] == other_region[j]:
                            included_in_other_region = True
                            
            if potential_region not in ai_regions_raw and not included_in_other_region:
                ai_regions.append([ai_good, clause, potential_region])
                ai_regions_raw.append(potential_region)
                region_name = ""
                k = 0
                for j in range(metadata_dimensions):
                    if j in clause:
                        region_name += dataset.metadata_labels[j] + ": "+ potential_region[k] +";" 
                        k += 1
                if ai_good:
                    region_name = "AI is good at: " + region_name
                else:
                    region_name = "AI is bad at: " + region_name

                ai_regions_names.append(region_name)
            if len(ai_regions) == N_CLAUSES:
                break
        # get human regions
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > 60:
                return None
            # sample a metadata point
            metadata_point = dataset.metadata[np.random.randint(len(dataset.metadata))]
            # sample CLAUSE_LENGTH metadata dimensions
            clause = np.random.choice(metadata_dimensions, CLAUSE_LENGTH, replace=False)
            # sort clause
            clause = np.sort(clause)
            potential_region = [metadata_point[i] for i in clause]
            # count number of points in region
            region_size = 0
            for i in range(len(dataset.metadata)):
                if all(dataset.metadata[i][j] == potential_region[k] for k,j in enumerate(clause)):
                    region_size += 1
            # check if region is too small or too big
            if region_size < MIN_REGION_SIZE*len(dataset.metadata) or region_size > MAX_REGION_SIZE*len(dataset.metadata):
                continue
            # check if region already exists
            hum_good = len(hum_regions) < N_CLAUSES/2
            # included in other region
            included_in_other_region = False
            for other_region in ai_regions_raw:
                # check crosswise inclusion
                for i in range(len(clause)):
                    for j in range(len(clause)):
                        if potential_region[i] == other_region[j]:
                            included_in_other_region = True
            for other_region in hum_regions_raw:
                # check crosswise inclusion
                for i in range(len(clause)):
                    for j in range(len(clause)):
                        if potential_region[i] == other_region[j]:
                            included_in_other_region = True
            if potential_region not in hum_regions_raw and potential_region not in ai_regions_raw and not included_in_other_region:
                hum_regions.append([hum_good, clause, potential_region])
                hum_regions_raw.append(potential_region)
                region_name = ""
                k = 0
                for j in range(metadata_dimensions):
                    if j in clause:
                        region_name += dataset.metadata_labels[j] + ": "+ potential_region[k] +"; "
                        k += 1
                if hum_good:
                    region_name = "Human is good at: " + region_name
                else:
                    region_name = "Human is bad at: " + region_name
                hum_regions_names.append(region_name)
            if len(hum_regions) == N_CLAUSES:
                break
        num_classes = len(np.unique(dataset.data_y))
        true_regions = np.zeros(len(dataset.data_y))
        for i in range(len(dataset.data_y)):
            any_region_satisfied = False
            for idx_region, region in enumerate(ai_regions):
                region_satisfied = all(dataset.metadata[i][j] == region[2][k] for k,j in enumerate(region[1]))
                # check if all other regions are not satisfied
                if region_satisfied:
                    any_region_satisfied = True
                    true_regions[i] = idx_region +1 

                    if region[0]:
                        # good region, sample coin
                        if np.random.rand() < GOOD_REGION_ACC:
                            ai_preds[i] = dataset.data_y[i]
                        else:
                            ai_preds[i] = np.random.choice(list(set(range(num_classes))-{dataset.data_y[i]}))
                    else:
                        if np.random.rand() < BAD_REGION_ACC:
                            ai_preds[i] = dataset.data_y[i]
                        else:
                            ai_preds[i] = np.random.choice(list(set(range(num_classes))-{dataset.data_y[i]}))
                    break
                
            if not any_region_satisfied:
                if np.random.rand() < OTHER_REGION_ACC:
                    ai_preds[i] = dataset.data_y[i]
                else:
                    ai_preds[i] = np.random.choice(list(set(range(num_classes))-{dataset.data_y[i]}))
        # get human predictions
        for i in range(len(dataset.data_y)):
            any_region_satisfied = False
            for idx_region, region in enumerate(hum_regions):
                region_satisfied = all(dataset.metadata[i][j] == region[2][k] for k,j in enumerate(region[1]))
                # check if all other regions are not satisfied
                if region_satisfied:
                        any_region_satisfied = True
                        true_regions[i] = idx_region +1 + len(ai_regions)

                        if region[0]:
                            # good region, sample coin
                            if np.random.rand() < GOOD_REGION_ACC:
                                hum_preds[i] = dataset.data_y[i]
                            else:
                                hum_preds[i] = np.random.choice(list(set(range(num_classes))-{dataset.data_y[i]}))
                        else:
                            if np.random.rand() < BAD_REGION_ACC:
                                hum_preds[i] = dataset.data_y[i]
                            else:
                                hum_preds[i] = np.random.choice(list(set(range(num_classes))-{dataset.data_y[i]}))
                        break
                    
                if not any_region_satisfied:
                    if np.random.rand() < OTHER_REGION_ACC:
                        hum_preds[i] = dataset.data_y[i]
                    else:
                        hum_preds[i] = np.random.choice(list(set(range(num_classes))-{dataset.data_y[i]}))
        return ai_preds, hum_preds, true_regions, ai_regions_names, hum_regions_names

    while True:
        result = get_synth_ai_human_regions_helper()
        if result is not None:
            return result
        print("timed out")
   