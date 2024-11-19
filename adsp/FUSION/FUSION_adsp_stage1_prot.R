library(BEDMatrix)
library(dplyr)
library(tidyr)
library(igraph)
library(uuid)
library(data.table)
library(glmnet)
library(caret)

#setwd("~/deepRIV/UKB/data")
source("/home/panwei/he000176/deepRIV/UKB/code/allele_qc.R")

# useful helper functions

generate_random_dir_name <- function() {
  return(paste0("dir_", UUIDgenerate()))
}

create_random_directory <- function(path = '/scratch.global/he000176/') {
  if (!dir.exists(path)) {
    dir.create(path)
  }
  
  dir_name <- paste0(path, '/', generate_random_dir_name(), '/')
  
  # Create the directory
  dir.create(dir_name)
  return(dir_name)
}

mean_imputation <- function(data) {
  impute_column <- function(col) {
    na_indices <- which(is.na(col))
    if (length(na_indices) > 0) {
      mean_value <- mean(col, na.rm = TRUE)
      col[na_indices] <- mean_value
    }
    return(col)
  }
  
  imputed_data <- apply(data, 2, impute_column)
  return(data.frame(imputed_data))
}

cv_stage1 <- function(snp, exp, method = c('lasso', 'elastic-net', 'both'),  
                      n_inner_folds = 5, n_outer_folds = 3, seed = 1){

  # Load necessary libraries
  library(glmnet)
  library(caret)
  
  set.seed(seed)
  
  # match sample ids
  idx = match(exp$IID, rownames(snp))
  snp = snp[idx,]
  cat(all(rownames(snp)==exp$IID),'\n')
  exp = exp$pheno
  
   
  # Ensure the method is valid
  method <- match.arg(method, c('lasso', 'elastic-net', 'both'))
  
  # Outer CV splitting
  outer_folds <- createFolds(exp, k = n_outer_folds, list = TRUE, returnTrain = TRUE)
  
  # Initialize lists to store predictions and coefficients
  all_preds <- list(lasso = rep(NA, length(exp)), elastic_net = rep(NA, length(exp)))
  results <- list(coefficients = list(), r2 = list())
  
  # Helper function to calculate out-of-sample R2
  calculate_r2 <- function(pred, actual) {
    cor(pred, actual)^2
  }
  
  # Loop over outer folds
  for (fold in 1:n_outer_folds) {
    print(fold)
    
    # Train/Test split for outer CV
    train_idx <- outer_folds[[fold]]
    test_idx <- setdiff(1:nrow(snp), train_idx)
    
    snp_train <- scale(snp[train_idx, , drop = FALSE])
    exp_train <- scale(exp[train_idx])
    snp_test <- scale(snp[test_idx, , drop = FALSE])
    exp_test <- scale(exp[test_idx])
    
    if (any(is.na(snp_train))){
      na_columns_train = which(apply(snp_train, 2, function(col){any(is.na(col))}))
      snp_train[, na_columns_train] = snp[train_idx, na_columns_train, drop = FALSE]
    }   

    if (any(is.na(snp_test))) {
      na_columns_test = which(apply(snp_test, 2, function(col){any(is.na(col))}))
      snp_test[, na_columns_test] = snp[test_idx, na_columns_test, drop = FALSE]
    }
      
      
    # Lasso
    if (method %in% c('lasso', 'both')) {
      lasso_model <- cv.glmnet(snp_train, exp_train, alpha = 1, nfolds = n_inner_folds)
      lasso_pred <- predict(lasso_model, newx = snp_test, s = "lambda.min")
      all_preds$lasso[test_idx] <- lasso_pred
    }
    
    # Elastic-Net
    if (method %in% c('elastic-net', 'both')) {
      elastic_net_model <- cv.glmnet(snp_train, exp_train, alpha = 0.5, nfolds = n_inner_folds)
      elastic_net_pred <- predict(elastic_net_model, newx = snp_test, s = "lambda.min")
      all_preds$elastic_net[test_idx] <- elastic_net_pred
    }
  }
  
  # Calculate the overall out-of-sample R2 for the concatenated predictions
  r2 <- list()
  if (method %in% c('lasso', 'both')) {
    r2$lasso <- calculate_r2(all_preds$lasso, exp)
  }
  if (method %in% c('elastic-net', 'both')) {
    r2$elastic_net <- calculate_r2(all_preds$elastic_net, exp)
  }
  
  # Now retune the model using the entire dataset and refit with the best parameters
  final_coefficients <- list()
  lambda_values <- list()
  
  # normalize data
  snp = scale(snp)
  exp = scale(exp)
                                   
  # Lasso on the entire dataset
  if (method %in% c('lasso', 'both')) {
    final_lasso_model <- cv.glmnet(snp, exp, alpha = 1, nfolds = n_inner_folds)
    final_coefficients$lasso <- coef(final_lasso_model, s = "lambda.min")[-1, ]  # Exclude intercept
    lambda_values$lasso <- final_lasso_model$lambda.min
  }
  
  # Elastic-Net on the entire dataset
  if (method %in% c('elastic-net', 'both')) {
    final_elastic_net_model <- cv.glmnet(snp, exp, alpha = 0.5, nfolds = n_inner_folds)
    final_coefficients$elastic_net <- coef(final_elastic_net_model, s = "lambda.min")[-1, ]  # Exclude intercept
    lambda_values$elastic_net <- final_elastic_net_model$lambda.min
  }
  
  # Combine the coefficients into a single data.frame
  coef_df <- data.frame(snp = colnames(snp))
  
  if (method %in% c('lasso', 'both')) {
    coef_df$lasso <- final_coefficients$lasso
  }
  if (method %in% c('elastic-net', 'both')) {
    coef_df$elastic_net <- final_coefficients$elastic_net
  }
  
  # Return coefficients and out-of-sample R2
  return(list(coefficients = coef_df, r2 = r2, lambda = lambda_values))
}


##  # Covariates
COV5TABLE = fread("/home/panwei/he000176/deepRIV/UKB/data/covariates_age_baseline.txt",header = T)
### proteomics
PROT = fread("/home/panwei/shared/UKBiobankIndiv/Olink/olink_data.txt")
whole_blood_buddy = '/home/panwei/shared/UKBiobankIndiv/WBA_plink_keep.txt'
annot = fread('/home/panwei/he000176/deepRIV/ImputedTraits/code/adsp/olink_annotations.tsv')
# remove the X chromosome
annot = annot[annot$chr_hg19 != 'X',]

### ukb phenotype
white_unrelated_keep = fread('/home/panwei/shared/UKBiobankIndiv/WBA_plink_keep.txt',
                             header = F, data.table = F)

### NEW ###
adsp_pheno_all = fread('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/adsp_cov.csv',
                       data.table = F)


args = commandArgs(trailingOnly = TRUE)
file_num = as.integer(args[1])
runs = 1
start_ind = (file_num-1)*runs + 1
# end_ind = min(file_num*runs, length(unique(annot$ukb_code)))


############### missing proteins (comment out when not running for missing genes) ###############
# results_folder = paste0("/home/panwei/he000176/deepRIV/ImputedTraits/results/adsp/prot/stage1/")
# files_in_folder = list.files(path = results_folder, pattern = "\\.RData$", full.names = FALSE)
# prot_in_folder = sub("\\.RData$", "", files_in_folder)  
# missing_idx = which(!(annot$gene_symbol %in% prot_in_folder))
# missing_annot = annot[missing_idx,]
# fwrite(missing_annot, "/home/panwei/he000176/deepRIV/ImputedTraits/results/adsp/prot/stage1/missing_annot.csv")

missing_annot = fread("/home/panwei/he000176/deepRIV/ImputedTraits/results/adsp/prot/stage1/missing_annot.csv")
end_ind = min(file_num*runs , nrow(missing_annot))

# for(gene_ind in missing_idx[start_ind:end_ind]){

for(prot_idx in start_ind:end_ind){
  cat('prot_idx: ',prot_idx,'\n')
  # get covariates
  cov5table = COV5TABLE
  # get gene expression
  
  # prot_inds = unique(annot$ukb_code)
  prot_inds = unique(missing_annot$ukb_code)
  prot_ind = prot_inds[prot_idx]

  prot_name = as.character(annot$uniprot_id[annot$ukb_code==prot_ind])
  if(length(prot_name)==0) next()
  gene_symbol = as.character(annot$gene_symbol[annot$ukb_code==prot_ind])
  prot_exp_path = paste0('/scratch.global/he000176/UKB/prot_exp_adj_cov/',gene_symbol,'.txt')
  if(!file.exists(prot_exp_path)) next()
  prot = fread(prot_exp_path)
  
  chr = as.integer(annot$chr_hg19[annot$ukb_code==prot_ind])
  start = floor(annot$start_bp_hg19[annot$ukb_code==prot_ind]/1000) - 500
  end = floor(annot$end_bp_hg19[annot$ukb_code==prot_ind]/1000) + 500
      
  cat('prot_ind: ',prot_ind,'; gene_symbol: ',gene_symbol,'; chr: ',chr,'\n')
  
  if(is.na(start) || is.na(end)){
    cat('NA position values! \n')
    next()
  }
    
  data_save_path = paste0('/scratch.global/he000176/ImputedTraits/stage1_data/prot/',gene_symbol,'.rds')
  results_path = '/home/panwei/he000176/deepRIV/ImputedTraits/results/adsp/prot/stage1/'
  
  # start analysis ----------------------------------------------------------
  ukb_pheno_all = fread('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/igap_imputed_ad.txt',data.table = F)
  ukb_igap = data.frame(f.eid = as.integer(ukb_pheno_all$FID), pheno = ukb_pheno_all$a40)

  ukb_pheno_all = fread('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/eadb_imputed_ad.txt',data.table = F)
  ukb_eadb = data.frame(f.eid = as.integer(ukb_pheno_all$FID), pheno = ukb_pheno_all$e40)

  ukb_pheno_all = fread('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/igap_imputed_ad.txt',data.table = F)
  ukb_proxy = data.frame(f.eid = as.integer(ukb_pheno_all$FID), pheno = ukb_pheno_all$testtrue - 1)

  adsp_pheno = data.frame(id = adsp_pheno_all$SampleID, pheno = adsp_pheno_all$DX_harmonized)
  adsp_pheno = na.omit(adsp_pheno)
  

  UKB_dir = create_random_directory('/scratch.global/he000176/UKB/tmp/')
  adsp_dir = create_random_directory('/scratch.global/he000176/ADSP/tmp/')
  print(UKB_dir)
  print(adsp_dir)
  
  ############ GENERATE --export-allele file ############
  prefix = paste0('/home/panwei/shared/UKBiobankIndiv/imputed/pgen/ukbb_chr',chr,'_1')
  tmp_bim = fread(paste0(prefix,'.pvar'))
  tmp_bim = data.frame(ID = tmp_bim$ID, REF = tmp_bim$REF)
  fwrite(tmp_bim, paste0(UKB_dir,'ukb_allele_map.txt'), sep=' ', col.names = F)
  rm(tmp_bim)
  #######################################################
    
  ukb_out = paste0(UKB_dir,prot_name)
  ukb_plink_command = paste0('plink2 --pfile ',prefix,
                             ' --chr ', chr," --from-kb ",start," --to-kb ",end,
                             ' --keep ',whole_blood_buddy,
                             ' --maf 0.05 --hwe 1e-5 1000 midp',
                             ' --export A include-alt',
                             ' --export-allele ', UKB_dir, 'ukb_allele_map.txt',
                             ' --snps-only just-acgt',
                             ' --out ',ukb_out)
  ukb_plink_msg = system(ukb_plink_command,intern=TRUE)
  
  ukb_snp = list()
  ukb_snp$bed = tryCatch(fread(paste0(ukb_out,'.raw'),data.table=F),
                     error=function(e){cat("ERROR :",
                                           conditionMessage(e),
                                           "\n")})
  if(length(ukb_snp)==0)
  {
    unlink(UKB_dir, recursive = TRUE)
    unlink(adsp_dir, recursive = TRUE)
    next()
  }

  ukb_snp$fam = ukb_snp$bed[,c(1:6)] 
  # create bim file without pos
  tmp_ukb_names = colnames(ukb_snp$bed)[-c(1:6)]
  ukb_snp$bim = data.frame(ID = sub("_.*", "", tmp_ukb_names), 
                           REF = sub(".*_(.*)\\(/.*", "\\1", tmp_ukb_names), 
                           ALT = sub(".*\\(/(.*)\\)", "\\1", tmp_ukb_names))
  # check for special characters in rs id, REF and ALT alleles
  unique(unlist(strsplit(paste(ukb_snp$bim$ID,collapse=''),split='')))
  unique(unlist(strsplit(paste(ukb_snp$bim$REF,collapse=''),split='')))
  unique(unlist(strsplit(paste(ukb_snp$bim$ALT,collapse=''),split='')))

  # turn ukb_snp$bed into an R matrix
  ukb_snp$bed = as.matrix(ukb_snp$bed[,-c(1:6)])
  colnames(ukb_snp$bed) = ukb_snp$bim$ID
  rownames(ukb_snp$bed) = ukb_snp$fam$IID
  rm(tmp_ukb_names)
  
  ########## UPDATED adsp snps TO BE COMPATIBLE WITH --export-allele ##########
  tmp_pos_file = data.frame(ID = ukb_snp$bim$ID, REF = ukb_snp$bim$REF)
  fwrite(tmp_pos_file, sep = ' ',
         paste0(adsp_dir,'/tmp_rs_file.txt'),col.names = F)
  #############################################################################
    
  adsp_snp_path = paste0('/scratch.global/he000176/ADSP/bed_hwe1e15/adsp_plink_chr',chr)
  adsp_out = paste0(adsp_dir,prot_name)
  adsp_plink_command = paste0('plink2 --bfile ', adsp_snp_path,
                             ' --chr ', chr,
                             ' --extract ',adsp_dir,'tmp_rs_file.txt',
                             ' --export A include-alt',
                             ' --export-allele ', adsp_dir, 'tmp_rs_file.txt',
                             ' --snps-only just-acgt',
                             ' --out ',adsp_out)
  adsp_plink_msg = system(adsp_plink_command,intern=TRUE)
  
  adsp_snp = list()
  adsp_snp$bed = tryCatch(fread(paste0(adsp_out,'.raw'), data.table=F),
                     error=function(e){cat("ERROR :",
                                           conditionMessage(e),
                                           "\n")})
  if(length(adsp_snp)==0)
  {
    unlink(UKB_dir, recursive = TRUE)
    unlink(adsp_dir, recursive = TRUE)
    next()
  }

  adsp_snp$fam = adsp_snp$bed[,c(1:6)] 
  # create bim file without pos
  tmp_adsp_names = colnames(adsp_snp$bed)[-c(1:6)]
  adsp_snp$bim = data.frame(ID = sub("_.*", "", tmp_adsp_names), 
                            REF = sub(".*_(.*)\\(/.*", "\\1", tmp_adsp_names), 
                            ALT = sub(".*\\(/(.*)\\)", "\\1", tmp_adsp_names))
  # check for special characters in rs id, REF and ALT alleles
  unique(unlist(strsplit(paste(adsp_snp$bim$ID,collapse=''),split='')))
  unique(unlist(strsplit(paste(adsp_snp$bim$REF,collapse=''),split='')))
  unique(unlist(strsplit(paste(adsp_snp$bim$ALT,collapse=''),split='')))

  # turn adsp_snp$bed into an R matrix
  adsp_snp$bed = as.matrix(adsp_snp$bed[,-c(1:6)])
  colnames(adsp_snp$bed) = adsp_snp$bim$ID
  rownames(adsp_snp$bed) = adsp_snp$fam$IID 
  rm(tmp_adsp_names)
  
  
  # remove ambiguous (palindromic) SNPs and flip alleles (ref, effect) for UKB
  snp_bim_ukb = merge(adsp_snp$bim,ukb_snp$bim,by=c("ID"))
  remove_flip = allele.qc(snp_bim_ukb$REF.x,snp_bim_ukb$ALT.x,
                          snp_bim_ukb$REF.y,snp_bim_ukb$ALT.y)
  print(paste0("any SNP's orientation needs to be flipped? ", any(remove_flip$flip[remove_flip$keep]))) # should be FALSE
  flip_snp = snp_bim_ukb$ID[remove_flip$flip]
  snp_bim_ukb = snp_bim_ukb[remove_flip$keep,]
  
  ukb_snp_bed = ukb_snp$bed[,which(colnames(ukb_snp$bed) %in% snp_bim_ukb$ID),drop=FALSE] 
  ukb_snp_bed[,which(colnames(ukb_snp_bed) %in% flip_snp)] = 2 - ukb_snp_bed[,which(colnames(ukb_snp_bed) %in% flip_snp)]
  
  adsp_snp_bed = adsp_snp$bed[,which(colnames(adsp_snp$bed) %in% snp_bim_ukb$ID),drop=FALSE] 

  ukb_bed_ind = match(colnames(adsp_snp_bed), colnames(ukb_snp_bed))
  ukb_snp_bed = ukb_snp_bed[, ukb_bed_ind, drop = FALSE]
  
  
  # match adsp_pheno with adsp_snp_bed
  ### NEW replace : with _ and only remove the leaning 0_ ###
  rownames(adsp_snp_bed) = sub("^0_", "", rownames(adsp_snp_bed))
  ###    
    
  keep_ukb_indiv = intersect(adsp_pheno$id,adsp_snp$fam$IID)
  adsp_pheno = adsp_pheno %>% filter(id %in% keep_ukb_indiv)
  adsp_snp_bed = adsp_snp_bed[which(rownames(adsp_snp_bed)%in%keep_ukb_indiv),,drop=FALSE]
  adsp_pheno = adsp_pheno[match(rownames(adsp_snp_bed),adsp_pheno$id),,drop=FALSE]
  
  # divide ukb_snp_bed into stage1 and stage2
  common_id = intersect(prot$IID, rownames(ukb_snp_bed))
  ukb_snp_beds1 = ukb_snp_bed[match(common_id, rownames(ukb_snp_bed)),,drop=FALSE]
  prot = prot[match(common_id, prot$IID),,drop=FALSE]
  
  ukb_snp_beds2 = ukb_snp_bed[!(rownames(ukb_snp_bed)%in%common_id),,drop=FALSE]
  
  # match covariates with the two ukb snp bed matrices
  # cov5table_s1 = cov5table[match(rownames(ukb_snp_beds1),cov5table$FID),,drop=FALSE]
  cov5table_s2 = cov5table[match(rownames(ukb_snp_beds2),cov5table$FID),,drop=FALSE]
  
  keep_ukb_indiv = intersect(ukb_igap$f.eid,rownames(ukb_snp_beds2))
  ukb_igap = ukb_igap %>% filter(f.eid %in% keep_ukb_indiv)
  ukb_eadb = ukb_eadb %>% filter(f.eid %in% keep_ukb_indiv)
  ukb_proxy = ukb_proxy %>% filter(f.eid %in% keep_ukb_indiv)
  
  ukb_snp_beds2 = ukb_snp_beds2[which(rownames(ukb_snp_beds2)%in%keep_ukb_indiv),,drop=FALSE]
  
  ukb_igap = ukb_igap[match(rownames(ukb_snp_beds2),ukb_igap$f.eid),,drop=FALSE]
  ukb_eadb = ukb_eadb[match(rownames(ukb_snp_beds2),ukb_eadb$f.eid),,drop=FALSE]
  ukb_proxy = ukb_proxy[match(rownames(ukb_snp_beds2),ukb_proxy$f.eid),,drop=FALSE]
  
  cov5table_s2 = cov5table_s2[match(rownames(ukb_snp_beds2),cov5table_s2$FID),,drop=FALSE]
  ukb_snp_beds1 = mean_imputation(ukb_snp_beds1)
  
  if(nrow(ukb_igap)< 3){
    unlink(UKB_dir, recursive = TRUE)
    unlink(adsp_dir, recursive = TRUE)
    next()
  }
  if(ncol(ukb_snp_beds2) < 2)
  {
    unlink(UKB_dir, recursive = TRUE)
    unlink(adsp_dir, recursive = TRUE)
    next()
  }
  
  num_snps_common = ncol(adsp_snp_bed)
  
  #################################### continue after regressing protein exp on cov ###############################
  
  #### LASSO and elastic net
  cat('prot exp id and snp id match: ',all(prot$IID==as.integer(rownames(ukb_snp_beds1))),'\n')
  #tmp_start_time = Sys.time()
  res = cv_stage1(ukb_snp_beds1, prot, method='both')
  #tmp_end_time = Sys.time()
  #tmp_end_time - tmp_start_time
  
  # for TWAS-LQ
  prot2 = prot
  prot2$pheno = prot2$pheno^2
  res_x2 = cv_stage1(ukb_snp_beds1, prot2, method='both')

  ### UKB
  y_ukb_ID = as.integer(rownames(ukb_snp_beds2))
  # y_ukb_total = as.numeric(unlist(ukb_igap[,2]))
  rs_id = colnames(ukb_snp_beds2)
  y_adsp_ID = rownames(adsp_snp_bed)
  y_adsp_total = as.numeric(unlist(adsp_pheno[,2]))


  # get snps with non-zero coefficients
  lasso_idx = which(res$coefficients$lasso!=0)
  lasso_X2_idx = which(res_x2$coefficients$lasso!=0)
  elastic_idx = which(res$coefficients$elastic_net!=0)
  elastic_X2_idx = which(res_x2$coefficients$elastic_net!=0)

  nonzero_idx = sort(Reduce(union, list(lasso_idx, lasso_X2_idx, elastic_idx, elastic_X2_idx)))

  ukb_snp_beds1_reduced = ukb_snp_beds1[,nonzero_idx,drop=F]
  ukb_snp_beds2_reduced = ukb_snp_beds2[,nonzero_idx,drop=F]
  adsp_snp_bed_reduced = adsp_snp_bed[,nonzero_idx,drop=F]

  # get two versions of rs id, one for the original matrix, one for the matrix after removing zero coefs
  rs_id_reduced = rs_id[nonzero_idx] 
  

  if (TRUE){
    save(res, res_x2, num_snps_common, rs_id, prot_name, gene_symbol, chr, file = paste0(results_path,'/',gene_symbol,'.RData'))
  }
  
  # remove tmp dir
  unlink(UKB_dir, recursive = TRUE)
  unlink(adsp_dir, recursive = TRUE)
  
  FINAL_RESULT = list(protein_id=prot_name,
                      gene_symbol=gene_symbol,
                      chr=chr,
                      #ukb_snp_beds2 = ukb_snp_beds2,
                      ukb_snp_beds2_reduced = ukb_snp_beds2_reduced,
                      cov_s2 = cov5table_s2,
                      hatbetaX1 = res$coefficients[nonzero_idx,,drop=F],
                      hatbetaX2 = res_x2$coefficients[nonzero_idx,,drop=F],
                      y_ukb_ID = y_ukb_ID,
                      stage1_r2 = res$r2,
                      stage1_X2_r2 = res_x2$r2,
                      num_snps_common = num_snps_common,
                      rs_id = rs_id,
                      rs_id_reduced = rs_id_reduced,
                      adsp_snp_bed_reduced = adsp_snp_bed_reduced,
                      y_adsp_ID = y_adsp_ID,
                      y_adsp_total = y_adsp_total,
                      ukb_protein_sample_size = nrow(ukb_snp_beds1)
                      
  )
  
  saveRDS(FINAL_RESULT, file = data_save_path)  
}