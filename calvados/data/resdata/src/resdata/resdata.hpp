#ifndef _RESDATA_RESDATA_HPP
#define _RESDATA_RESDATA_HPP

// gromacs includes
#include <gromacs/trajectoryanalysis/topologyinformation.h>
// #include <gromacs/topology/mtop_util.h>
#include <gromacs/math/vec.h>
#include <gromacs/pbcutil/pbc.h>
#include <gromacs/fileio/tpxio.h>
#include <gromacs/fileio/confio.h>
#include <gromacs/topology/index.h>

// resdata includes
#include "io.hpp"
#include "indexing.hpp"
#include "parallel.hpp"
#include "density.hpp"
#include "mindist.hpp"
#include "xtc_frame.hpp"
#include "function_types.hpp"
#include "topology.hpp"
#include "lgmx.hpp"

// standard library imports
#include <iostream>
#include <omp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <memory>
#include <string>
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <fstream>
#include <chrono>
#include <tuple>

// xdrfile includes
#include <xdrfile.h>
#include <xdrfile_xtc.h>

// #define DEBUG
#ifdef DEBUG
#define LOG(x) std::cout << x << std::endl;
#define STRINGIFY(x) #x
#define SLOG(x) std::cout << STRINGIFY(x) << std::endl;
#else
#define LOG(x)
#define SLOG(x)
#endif

namespace resdata
{
  class RESData
  {
  private:
    std::string top_path_;
    std::string trj_path_;
    // general fields
    int n_x_;
    float dt_, t_begin_, t_end_;
    int nskip_;
    resdata::topology::Topology top_;
    rvec *xcm_;
    rvec **res_xcm_;
    float cutoff_, mol_cutoff_, mcut2_, cut_sig_2_;

    std::string index_path_;

    // molecule number fields
    bool simple_topology_;
    bool use_index_ = false;

    // weights fields
    float weights_sum_ = 0.0;
    std::string weights_path_;
    std::vector<float> weights_;

    // pbc fields
    bool no_pbc_;

    // frame fields
    resdata::xtc::Frame *frame_;
    XDRFILE *trj_;

    // density fields
    float dx_;
    std::size_t n_bins_;
    std::vector<float> density_bins_;

    using cmatrix = std::vector<std::vector<std::vector<std::vector<float>>>>;
    cmatrix intram_mat_density_;
    cmatrix interm_same_mat_density_;
    cmatrix interm_cross_mat_density_;
    cmatrix interm_same_maxcdf_mol_;
    cmatrix interm_cross_maxcdf_mol_;

    // temporary containers for maxcdf operations
    std::vector<std::vector<float>> frame_same_mat_;
    std::vector<std::vector<float>> frame_cross_mat_;
    std::vector<std::vector<std::mutex>> frame_same_mutex_;
    std::vector<std::vector<std::mutex>> frame_cross_mutex_;

    // parallelization fields
    int num_threads_;
    int num_mol_threads_;
    std::vector<std::thread> threads_;
    std::vector<std::thread> mol_threads_;
    resdata::parallel::Semaphore semaphore_;
    std::vector<resdata::indexing::SameThreadIndices> same_thread_indices_;
    std::vector<resdata::indexing::CrossThreadIndices> cross_thread_indices_;

    // mode selection, booleans and functions
    std::string mode_;
    bool intra_ = false, same_ = false, cross_ = false;

    // function types
    // using ftype_intra_ = resdata::ftypes::function_traits<decltype(&resdata::density::intra_mol_routine)>;
    // using ftype_same_ = resdata::ftypes::function_traits<decltype(&resdata::density::inter_mol_same_routine)>;
    // using ftype_cross_ = resdata::ftypes::function_traits<decltype(&resdata::density::inter_mol_cross_routine)>;

    // std::function<ftype_intra_::signature> f_intra_mol_;
    // std::function<ftype_same_::signature> f_inter_mol_same_;
    // std::function<ftype_cross_::signature> f_inter_mol_cross_;

    static void molecule_routine(
        const int i, const resdata::topology::Topology &top, rvec *xcm,
        float mcut2, float cut_sig_2,
        const std::vector<float> &density_bins, rvec **group_coms, float weight,
        std::vector<std::vector<float>> &frame_same_mat, std::vector<std::vector<float>> &frame_cross_mat,
        std::vector<std::vector<std::mutex>> &frame_same_mutex, std::vector<std::vector<std::mutex>> &frame_cross_mutex,
        cmatrix &intram_mat_density, cmatrix &interm_same_mat_density, cmatrix &interm_cross_mat_density,
        const bool intra, const bool same, const bool cross, resdata::parallel::Semaphore &semaphore)
    {
      semaphore.acquire();
      const std::vector<float> inv_num_mol = top.get_inv_num_mol();
      const std::vector<int> natmol2 = top.get_res_per_molecule();
      const std::vector<int> mol_id = top.mol_id();

      // int tmp_i = 0;
      // std::size_t mol_i = i, mol_j = 0;
      // const std::vector<int> mol_id = top_.mol_id();
      const int access_i = mol_id[i];

      // while (static_cast<int>(mol_i) - top_.get_n_mols(tmp_i) >= 0)
      // {
      //   mol_i -= top_.get_n_mols(tmp_i);
      //   tmp_i++;
      //   if (tmp_i == top_.get_n_mols())
      //     break;
      // }
      // if (mol_i == top_.get_n_mols(access_i))
      //   mol_i = 0;

      // int molb = 0;
      for (int j = 0; j < top.get_n_mols(); j++) // check if can be set to 0 => normalization
      {
        const int access_j = top.mol_id(j);
        if ((same || intra) && (!cross))
        {
          if (access_i != access_j)
            continue;
        }
        if ((!same && !intra) && cross)
        {
          if (access_i == access_j)
            continue;
        }
        if ((!same && !intra && !cross))
          continue;
        // if (j != 0)
        // if (mol_j == top_.get_n_mols(top_.mol_id()[j - 1]))
        // mol_j = 0;

        if (i != j)
        {
          rvec dx;
          if (top.get_pbc() != nullptr)
            pbc_dx(top.get_pbc(), xcm[i], xcm[j], dx);
          else
          {
            std::cout << "No PBC, using no PBC" << std::endl;
            rvec_sub(xcm[i], xcm[j], dx);
          }
          float dx2 = iprod(dx, dx);

          if (dx2 > mcut2)
            continue;
        }
        if (access_i != access_j && j < i)
          continue;

        // for (std::size_t ii = molb[i].front(); ii < molb[i].back() + 1; ii++)
        for (std::size_t gi = 0; gi < top.get_res_per_molecule(i, true); ++gi)
        {
          // for (std::size_t jj = molb[j].front(); jj < molb[j].back() + 1; jj++)
          for (std::size_t gj = 0; gj < top.get_res_per_molecule(j, true); ++gj)
          {
            int delta = gi - gj;
            rvec sym_dx;
            if (top.get_pbc() != nullptr)
              // pbc_dx(top_.get_pbc(), x[ii], x[jj], sym_dx);
              pbc_dx(top.get_pbc(), group_coms[i][gi], group_coms[j][gj], sym_dx);
            else
            {
              std::cout << "No PBC, using no PBC" << std::endl;
              rvec_sub(xcm[i], xcm[j], sym_dx);
            }
            float dx2 = iprod(sym_dx, sym_dx);

            if (i == j)
            {
              if (intra)
              {
                if (dx2 < cut_sig_2)
                {
                  resdata::density::intra_mol_routine(
                      i, gi, gj, dx2, weight, mol_id, natmol2, density_bins, inv_num_mol, frame_same_mutex, intram_mat_density);
                }
              }
            }
            else
            {
              if (access_i == access_j)
              {
                if (same)
                {
                  if (dx2 < cut_sig_2)
                  {
                    resdata::density::inter_mol_same_routine(
                        i, gi, gj, dx2, weight, mol_id, natmol2, density_bins, frame_same_mutex, frame_same_mat, interm_same_mat_density);
                  }
                  if (delta != 0)
                  {
                    if (top.get_pbc() != nullptr)
                      pbc_dx(top.get_pbc(), group_coms[i][gi - delta], group_coms[j][gj + delta], sym_dx);
                    else
                    {
                      std::cout << "No PBC, using no PBC" << std::endl;
                      rvec_sub(group_coms[i][gi - delta], group_coms[j][gj + delta], sym_dx);
                    }
                    dx2 = iprod(sym_dx, sym_dx);
                    if (dx2 < cut_sig_2)
                    {
                      resdata::density::inter_mol_same_routine(
                          i, gi, gj, dx2, weight, mol_id, natmol2, density_bins, frame_same_mutex, frame_same_mat, interm_same_mat_density);
                    }
                  }
                }
              }
              else
              {
                if (dx2 < cut_sig_2 && cross)
                {
                  resdata::density::inter_mol_cross_routine(
                      i, j, gi, gj, dx2, weight, mol_id, natmol2, top.get_cross_index(), density_bins, frame_cross_mutex, frame_cross_mat, interm_cross_mat_density);
                }
              }
            }
          }
        }
      }
      semaphore.release();
    }

  public:
    RESData(
        const std::string &top_path, const std::string &traj_path,
        float cutoff, float mol_cutoff, int nskip, int num_threads, int num_mol_threads,
        int dt, const std::string &mode, const std::string &weights_path,
        bool no_pbc, float t_begin, float t_end, const std::string &index_path, bool simple_topology) : cutoff_(cutoff), mol_cutoff_(mol_cutoff), nskip_(nskip), num_threads_(num_threads), num_mol_threads_(num_mol_threads),
                                                                                                        mode_(mode), weights_path_(weights_path), simple_topology_(simple_topology), index_path_(index_path),
                                                                                                        no_pbc_(no_pbc), dt_(dt), t_begin_(t_begin), t_end_(t_end), top_path_(top_path), trj_path_(traj_path)
    {
      mcut2_ = mol_cutoff_ * mol_cutoff_;
      cut_sig_2_ = (cutoff_) * (cutoff_);

      initAnalysis();
    }

    ~RESData()
    {
      free(frame_->x);
      free(frame_->offsets);
      free(frame_);
      free(xcm_);
      for (int i = 0; i < top_.get_n_mols(); i++)
      {
        free(res_xcm_[i]);
      }
      free(res_xcm_);
    }

    void initAnalysis()
    /**
     * @brief Initializes the analysis by setting up the molecule partitioning and the mode selection
     *
     * @todo Check if molecule block is empty
     */
    {
      LOG("Initializing analysis");
      bool bTop_;
      // declarations
      std::vector<int> index_;

      // read the index file
      if (std::filesystem::exists(std::filesystem::path(index_path_)))
      {
        throw std::runtime_error("Index file not implemented yet");
        // TODO read the index and then parse and then apply
        // int atm_i = 0;
        // for ( int &molb_i : top_.get_molblock_indices() )
        // {
        //   // int atom_n = mtop_->moltype[molb.type].atoms.nr;
        //   int atom_n = top_.get_molblocks()[molb_i].size();
        //   for ( int i = 0; i < atom_n ; i++)
        //   {
        //     if (atom_n == 0) continue;
        //     index_.push_back(atm_i);
        //     atm_i++;
        //   }
        // }
      }

      // matrix boxtop_;
      if (simple_topology_)
      {
        LOG("Reading simple topology file " << top_path_);
        auto stopol = resdata::io::read_simple_topology(top_path_);
        std::vector<int> n_mol = std::get<2>(stopol);
        for (int mi = 0; mi < std::get<2>(stopol).size(); mi++)
        {
          const std::string molecule_name = std::get<3>(stopol)[mi];
          const std::vector<std::string> atom_names = std::get<4>(stopol)[mi];
          const std::vector<std::string> residue_names = std::get<5>(stopol)[mi];
          const std::vector<int> residue_indices = std::get<6>(stopol)[mi];
          for (int nm = 0; nm < n_mol[mi]; nm++)
          {
            top_.add_molecule(molecule_name, atom_names, residue_names, residue_indices);
          }
        }
        PbcType pbcType = resdata::dtypes::pbc_type_map[std::get<1>(stopol)];
        matrix box;
        for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            box[i][j] = std::get<0>(stopol)[i][j];
          }
        }
        top_.set_topol_pbc(pbcType, box);
      }
      else
      {
        LOG("Reading topology file " << top_path_);
        gmx_mtop_t *mtop = static_cast<gmx_mtop_t *>(malloc(sizeof(gmx_mtop_t)));
        int natoms;
        matrix boxtop;

        PbcType pbcType = read_tpx(top_path_.c_str(), nullptr, boxtop, &natoms, nullptr, nullptr, mtop);
        top_.set_topol_pbc(pbcType, boxtop);

        LOG("Entering loop over mtop->molblock");
        for (const gmx_molblock_t &molb : mtop->molblock)
        {
          LOG("Processing molblock with nmol: " << molb.nmol << ", type: " << molb.type);
          for (int i = 0; i < molb.nmol; i++)
          {
            LOG("Processing molecule " << i << " of type " << molb.type);
            int natm_per_mol = mtop->moltype[molb.type].atoms.nr;
            LOG("Number of atoms per molecule: " << natm_per_mol);
            std::string molecule_name = std::string(mtop->moltype[molb.type].name[molb.type]); // todo check if correct access
            LOG("Molecule name: " << molecule_name);
            std::vector<std::string> atom_names;
            std::vector<std::string> res_names;
            std::vector<int> res_id;
            t_resinfo *resinfo = mtop->moltype[molb.type].atoms.resinfo;
            char ***atomtype = mtop->moltype[molb.type].atoms.atomname;
            LOG("Entering loop over atoms in molecule");
            for (int j = 0; j < natm_per_mol; j++)
            {
              LOG("Processing atom " << j);
              int resi = mtop->moltype[molb.type].atoms.atom[j].resind;
              LOG("Residue index: " << resi);
              atom_names.push_back(*atomtype[j]);
              LOG("Atom name: " << *atomtype[j]);
              res_names.push_back(*resinfo[resi].name);
              LOG("Residue name: " << *resinfo[resi].name);
              res_id.push_back(resi);
              LOG("Added residue index: " << resi);
            }
            top_.add_molecule(molecule_name, atom_names, res_names, res_id);
            LOG("Finished processing molecule " << i);
          }
        }
        LOG("Finished loop over mtop->molblock");
        LOG("done reading topoloy file");
        free(mtop);
        LOG("Freed mtop");
      }
      // exit(1);
      LOG("Filled index array")
      LOG("Applying index");
      top_.apply_index(index_);
      LOG("done applying index");

      // todo all the vectors should be set https://github.com/multi-ego/multi-eGO/blob/422408d37e5a2455dd24ad9b6f53ecb7c6e396ed/tools/resdata/src/resdata/resdata.hpp#L393
      // if (no_pbc_)
      //   top_.unset_pbc();

      int natom;
      long unsigned int nframe;
      int64_t *offsets;

      frame_ = (resdata::xtc::Frame *)malloc(sizeof(resdata::xtc::Frame));
      std::cout << "Reading trajectory file " << trj_path_ << std::endl;
      read_xtc_header(trj_path_.c_str(), &natom, &nframe, &offsets);
      *frame_ = resdata::xtc::Frame(natom);
      frame_->nframe = nframe;
      frame_->offsets = offsets;

      trj_ = xdrfile_open(trj_path_.c_str(), "r");

      // define index
      n_x_ = 0;
      xcm_ = (rvec *)malloc(top_.get_n_mols() * sizeof(rvec));
      // xcm_.resize(top_.get_n_mols());

      // CREATE VECTOR of RESIDUE indeces to map atom to residue
      printf("\nAssigning residue index to atoms\n");

      printf("Evaluating mode selection:\n");
      std::string tmp_mode;
      std::stringstream modestream{mode_};
      while (std::getline(modestream, tmp_mode, '+'))
      {
        if (tmp_mode == std::string("intra"))
        {
          intra_ = true;
        }
        else if (tmp_mode == std::string("same"))
        {
          same_ = true;
        }
        else if (tmp_mode == std::string("cross"))
        {
          cross_ = true;
        }
        else
        {
          printf("Wrong mode: %s\nMode must be one from: intra, same, cross. Use + to concatenate more than one, i.e. intra+cross\n", tmp_mode.c_str());
          exit(1);
        }
        printf(" - found %s\n", tmp_mode.c_str());
      }

      printf("\nNumber of different molecules %lu\n", top_.get_n_moltypes());
      bool check_same = false;
      for (std::size_t i = 0; i < top_.get_n_moltypes(); i++)
      {
        printf("    mol id: %lu, num mols: %u, size: %u,  nresidues:%u\n", i, top_.get_n_mols(i), top_.get_n_atoms_per_molecule()[i], top_.get_res_per_molecule(i));
        if (top_.get_n_mols(i) > 1)
          check_same = true;
      }

      if (num_threads_ > std::thread::hardware_concurrency())
      {
        num_threads_ = std::thread::hardware_concurrency();
        std::cout << "Maximum thread number surpassed. Scaling num_threads down to " << num_threads_ << std::endl;
      }
      if (num_mol_threads_ > std::thread::hardware_concurrency())
      {
        num_mol_threads_ = std::thread::hardware_concurrency();
        std::cout << "Maximum thread number surpassed. Scaling num_mol_threads down to " << num_mol_threads_ << std::endl;
      }
      if (num_mol_threads_ > top_.get_n_mols())
      {
        num_mol_threads_ = top_.get_n_mols();
        std::cout << "Number of molecule threads surpassed number of molecules. Setting num_mol_threads to " << num_mol_threads_ << std::endl;
      }
      threads_.resize(num_threads_);
      mol_threads_.resize(top_.get_n_mols());
      semaphore_.set_counter(num_mol_threads_);
      std::cout << "Using " << num_threads_ << " threads and " << num_mol_threads_ << " molecule threads" << std::endl;

      printf("\nActivating modes\n");
      if (!check_same && same_)
        same_ = false;
      if (cross_ && top_.get_n_moltypes() < 2)
      {
        cross_ = false;
        printf(":: deactivating cross mode (only 1 type of molecule found)\n");
      }

      if (top_.get_n_mols() > 1 && (same_ || cross_))
      {
        printf("\n\n::::::::::::WARNING::::::::::::\nMore than 1 molcule found in the system.\nFix pbc before running resdata using pbc mol\n");
        printf(":::::::::::::::::::::::::::::::\n\n");
      }

      // res_xcm_.resize(top_.get_n_mols());
      res_xcm_ = (rvec **)malloc(top_.get_n_mols() * sizeof(rvec *));
      for (int i = 0; i < top_.get_n_mols(); i++)
      {
        res_xcm_[i] = (rvec *)malloc(top_.get_res_per_molecule(top_.mol_id(i)) * sizeof(rvec));
      }
      // for (int i = 0; i < top_.get_n_mols(); i++)
      // {
      // res_xcm_[i].resize(top_.get_res_per_molecule(i));
      // }

      if (same_)
      {
        std::cout << ":: activating intermat same calculations" << std::endl;
        // f_inter_mol_same_ = resdata::density::inter_mol_same_routine;
        interm_same_mat_density_.resize(top_.get_n_moltypes());
        interm_same_maxcdf_mol_.resize(top_.get_n_moltypes());
      }
      if (cross_)
      {
        std::cout << ":: activating intermat cross calculations" << std::endl;
        // f_inter_mol_cross_ = resdata::density::inter_mol_cross_routine;
        interm_cross_mat_density_.resize((top_.get_n_moltypes() * (top_.get_n_moltypes() - 1)) / 2);
        interm_cross_maxcdf_mol_.resize((top_.get_n_moltypes() * (top_.get_n_moltypes() - 1)) / 2);
      }
      if (intra_)
      {
        std::cout << ":: activating intramat calculations" << std::endl;
        // f_intra_mol_ = resdata::density::intra_mol_routine;
        intram_mat_density_.resize(top_.get_n_moltypes());
      }

      density_bins_.resize(resdata::indexing::n_bins(cutoff_));
      for (std::size_t i = 0; i < density_bins_.size(); i++)
        density_bins_[i] = cutoff_ / static_cast<float>(density_bins_.size()) * static_cast<float>(i) + cutoff_ / static_cast<float>(density_bins_.size() * 2);

      // int cross_count = 0;
      // if (cross_)
      // cross_index_.resize(top_.get_n_moltypes(), std::vector<int>(top_.get_n_moltypes(), 0));
      for (std::size_t i = 0; i < top_.get_n_moltypes(); i++)
      {
        if (same_)
        {
          interm_same_mat_density_[i].resize(top_.get_res_per_molecule(i), std::vector<std::vector<float>>(top_.get_res_per_molecule(i), std::vector<float>(resdata::indexing::n_bins(cutoff_), 0)));
          interm_same_maxcdf_mol_[i].resize(top_.get_res_per_molecule(i), std::vector<std::vector<float>>(top_.get_res_per_molecule(i), std::vector<float>(resdata::indexing::n_bins(cutoff_), 0)));
        }
        if (intra_)
        {
          intram_mat_density_[i].resize(top_.get_res_per_molecule(i), std::vector<std::vector<float>>(top_.get_res_per_molecule(i), std::vector<float>(resdata::indexing::n_bins(cutoff_), 0)));
        }
        for (std::size_t j = i + 1; j < top_.get_n_moltypes() && cross_; j++)
        {
          int cross_index = top_.get_cross_index(i, j);
          interm_cross_mat_density_[cross_index].resize(top_.get_res_per_molecule(i), std::vector<std::vector<float>>(top_.get_res_per_molecule(j), std::vector<float>(resdata::indexing::n_bins(cutoff_), 0)));
          interm_cross_maxcdf_mol_[cross_index].resize(top_.get_res_per_molecule(i), std::vector<std::vector<float>>(top_.get_res_per_molecule(j), std::vector<float>(resdata::indexing::n_bins(cutoff_), 0)));
          // cross_index_[i][j] = cross_count;
          // cross_count++;
        }
      }

      n_bins_ = resdata::indexing::n_bins(cutoff_);
      dx_ = cutoff_ / static_cast<float>(n_bins_);

      if (same_)
        frame_same_mat_.resize(top_.get_n_moltypes());
      if (intra_ || same_)
        frame_same_mutex_.resize(top_.get_n_moltypes());
      if (cross_)
        frame_cross_mat_.resize(top_.get_cross_index().size());
      if (cross_)
        frame_cross_mutex_.resize(top_.get_cross_index().size());
      for (std::size_t i = 0; i < top_.get_n_moltypes(); i++)
      {
        if (same_)
          frame_same_mat_[i].resize(top_.get_res_per_molecule(i) * top_.get_res_per_molecule(i) * top_.get_n_mols(i), 0);
        if (intra_ || same_)
          frame_same_mutex_[i] = std::vector<std::mutex>(top_.get_res_per_molecule(i) * top_.get_res_per_molecule(i));
        for (std::size_t j = i + 1; j < top_.get_n_moltypes() && cross_; j++)
        {
          frame_cross_mat_[top_.get_cross_index(i, j)].resize(top_.get_res_per_molecule(i) * top_.get_res_per_molecule(j) * top_.get_n_mols(i) * top_.get_n_mols(j), 0);
          frame_cross_mutex_[top_.get_cross_index(i, j)] = std::vector<std::mutex>(top_.get_res_per_molecule(i) * top_.get_res_per_molecule(j));
        }
      }

      if (weights_path_ != "")
      {
        printf("Weights file provided. Reading weights from %s\n", weights_path_.c_str());
        weights_ = resdata::io::read_weights_file(weights_path_);
        printf("Found %li frame weights in file\n", weights_.size());
        float w_sum = std::accumulate(std::begin(weights_), std::end(weights_), 0.0, std::plus<>());
        printf("Sum of weights amounts to %lf\n", w_sum);
        // weights_sum_ = 0.;
      }
      std::cout << "Calculating threading indices" << std::endl;
      /* calculate the mindist accumulation indices */
      LOG("Calculating same and cross thread indices");
      std::size_t num_ops_same = 0;
      LOG("NMOLS 0: " << top_.get_n_mols(0));
      for (std::size_t im = 0; im < top_.get_n_moltypes(); im++)
      {
        LOG("Calculating num_ops_same for molecule type " << im);
        num_ops_same += top_.get_n_mols(im) * (top_.get_res_per_molecule(im) * (top_.get_res_per_molecule(im) + 1)) / 2;
      }
      LOG("Total num_ops_same: " << num_ops_same);

      int n_per_thread_same = (same_) ? num_ops_same / num_threads_ : 0;
      int n_threads_same_uneven = (same_) ? num_ops_same % num_threads_ : 0;
      LOG("n_per_thread_same: " << n_per_thread_same << ", n_threads_same_uneven: " << n_threads_same_uneven);

      std::size_t start_mti_same = 0, start_im_same = 0, end_mti_same = 1, end_im_same = 1;
      std::size_t start_i_same = 0, start_j_same = 0, end_i_same = 0, end_j_same = 0;

      int num_ops_cross = 0;
      for (std::size_t im = 0; im < top_.get_n_moltypes(); im++)
      {
        for (std::size_t jm = im + 1; jm < top_.get_n_moltypes(); jm++)
        {
          LOG("Calculating num_ops_cross for molecule types " << im << " and " << jm);
          num_ops_cross += top_.get_n_mols(im) * top_.get_res_per_molecule(im) * top_.get_n_mols(jm) * top_.get_res_per_molecule(jm);
        }
      }
      LOG("Total num_ops_cross: " << num_ops_cross);

      int n_per_thread_cross = (cross_) ? num_ops_cross / num_threads_ : 0;
      int n_threads_cross_uneven = (cross_) ? num_ops_cross % num_threads_ : 0;
      LOG("n_per_thread_cross: " << n_per_thread_cross << ", n_threads_cross_uneven: " << n_threads_cross_uneven);

      std::size_t start_mti_cross = 0, start_mtj_cross = 1, start_im_cross = 0, start_jm_cross = 0, start_i_cross = 0, start_j_cross = 0;
      std::size_t end_mti_cross = 1, end_mtj_cross = 2, end_im_cross = 1, end_jm_cross = 1, end_i_cross = 0, end_j_cross = 0;

      for (int tid = 0; tid < num_threads_; tid++)
      {
        LOG("Processing thread " << tid);

        /* calculate same indices */
        int n_loop_operations_same = n_per_thread_same + (tid < n_threads_same_uneven ? 1 : 0);
        LOG("n_loop_operations_same: " << n_loop_operations_same);

        long int n_loop_operations_total_same = n_loop_operations_same;
        while (top_.get_res_per_molecule(end_mti_same - 1) - static_cast<int>(end_j_same) <= n_loop_operations_same)
        {
          LOG("Inside same loop: end_mti_same=" << end_mti_same << ", end_im_same=" << end_im_same
                                                << ", end_i_same=" << end_i_same << ", end_j_same=" << end_j_same);

          int sub_same = top_.get_res_per_molecule(end_mti_same - 1) - static_cast<int>(end_j_same);
          LOG("sub_same: " << sub_same);
          n_loop_operations_same -= sub_same;
          end_i_same++;
          end_j_same = end_i_same;

          if (static_cast<int>(end_j_same) == top_.get_res_per_molecule(end_mti_same - 1))
          {
            LOG("Incrementing end_im_same");
            end_im_same++;
            end_i_same = 0;
            end_j_same = 0;
          }
          if (static_cast<int>(end_im_same) - 1 == top_.get_n_mols(end_mti_same - 1))
          {
            LOG("Incrementing end_mti_same");
            end_mti_same++;
            end_im_same = 1;
            end_i_same = 0;
            end_j_same = 0;
          }
          if (n_loop_operations_same == 0)
            break;
        }
        end_j_same += n_loop_operations_same;

        /* calculate cross indices */
        int n_loop_operations_total_cross = n_per_thread_cross + (tid < n_threads_cross_uneven ? 1 : 0);
        LOG("n_loop_operations_total_cross: " << n_loop_operations_total_cross);

        if (top_.get_n_moltypes() > 1)
        {
          int n_loop_operations_cross = n_loop_operations_total_cross;
          while (top_.get_res_per_molecule(end_mti_cross - 1) * top_.get_res_per_molecule(end_mtj_cross - 1) -
                     (top_.get_res_per_molecule(end_mtj_cross - 1) * static_cast<int>(end_i_cross) + static_cast<int>(end_j_cross)) <=
                 n_loop_operations_cross)
          {
            LOG("Inside cross loop: end_mti_cross=" << end_mti_cross << ", end_mtj_cross=" << end_mtj_cross
                                                    << ", end_im_cross=" << end_im_cross << ", end_jm_cross=" << end_jm_cross
                                                    << ", end_i_cross=" << end_i_cross << ", end_j_cross=" << end_j_cross);

            int sub_cross = top_.get_res_per_molecule(end_mti_cross - 1) * top_.get_res_per_molecule(end_mtj_cross - 1) -
                            (top_.get_res_per_molecule(end_mtj_cross - 1) * static_cast<int>(end_i_cross) + static_cast<int>(end_j_cross));
            LOG("sub_cross: " << sub_cross);
            n_loop_operations_cross -= sub_cross;

            end_jm_cross++;
            end_i_cross = 0;
            end_j_cross = 0;

            if (end_jm_cross > top_.get_n_mols(end_mtj_cross - 1))
            {
              LOG("Incrementing end_mtj_cross");
              end_mtj_cross++;
              end_jm_cross = 1;
              end_i_cross = 0;
              end_j_cross = 0;
            }
            if (end_mtj_cross > top_.get_n_moltypes())
            {
              LOG("Incrementing end_im_cross");
              end_im_cross++;
              end_mtj_cross = end_mti_cross + 1;
              end_jm_cross = 1;
              end_i_cross = 0;
              end_j_cross = 0;
            }
            if (end_im_cross > top_.get_n_mols(end_mti_cross - 1))
            {
              LOG("Incrementing end_mti_cross");
              end_mti_cross++;
              end_mtj_cross = end_mti_cross + 1;
              end_im_cross = 1;
              end_jm_cross = 1;
              end_i_cross = 0;
              end_j_cross = 0;
            }
            if (end_mti_cross == top_.get_n_moltypes())
              break;
            if (n_loop_operations_cross == 0)
              break;
          }

          if (end_mti_cross < top_.get_n_moltypes())
          {
            end_i_cross += n_loop_operations_cross / top_.get_res_per_molecule(end_mtj_cross - 1);
            end_j_cross += n_loop_operations_cross % top_.get_res_per_molecule(end_mtj_cross - 1);
            end_i_cross += end_j_cross / top_.get_res_per_molecule(end_mtj_cross - 1);
            end_j_cross %= top_.get_res_per_molecule(end_mtj_cross - 1);
          }
        }

        if (same_)
        {
          LOG("Adding same_thread_indices_ for thread " << tid);
          same_thread_indices_.push_back({start_mti_same, start_im_same, start_i_same, start_j_same, end_mti_same,
                                          end_im_same, end_i_same, end_j_same, n_loop_operations_total_same});
        }
        else
        {
          same_thread_indices_.push_back({0, 0, 0, 0, 0, 0, 0, 0, 0});
        }

        if (cross_ && top_.get_n_moltypes() > 1)
        {
          LOG("Adding cross_thread_indices_ for thread " << tid);
          cross_thread_indices_.push_back({start_mti_cross, start_mtj_cross, start_im_cross, start_jm_cross, start_i_cross,
                                           start_j_cross, end_mti_cross, end_mtj_cross, end_im_cross, end_jm_cross, end_i_cross,
                                           end_j_cross, n_loop_operations_total_cross});
        }
        else
        {
          cross_thread_indices_.push_back({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
        }

        /* set new starts */
        start_mti_same = end_mti_same - 1;
        start_im_same = end_im_same - 1;
        start_i_same = end_i_same;
        start_j_same = end_j_same;

        start_mti_cross = end_mti_cross - 1;
        start_mtj_cross = end_mtj_cross - 1;
        start_im_cross = end_im_cross - 1;
        start_jm_cross = end_jm_cross - 1;
        start_i_cross = end_i_cross;
        start_j_cross = end_j_cross;
      }

      printf("Finished preprocessing. Starting frame-by-frame analysis.\n");
    }

    void run()
    {
      std::chrono::steady_clock::time_point begin;
      std::chrono::steady_clock::time_point end;
      std::cout << "Running frame-by-frame analysis" << std::endl;
      int frnr = 0;
      float progress = 0.0, new_progress = 0.0;
      resdata::io::print_progress_bar(progress);

      const std::vector<int> groups_per_molecule = top_.get_res_per_molecule();
      const std::vector<int> n_moltypes = top_.get_unique_molecules();
      const std::vector<std::vector<int>> cross_index = top_.get_cross_index();

      while (frame_->read_next_frame(trj_, no_pbc_, top_.get_pbc_type(), top_.get_pbc()) == exdrOK)
      {
        new_progress = static_cast<float>(frnr) / static_cast<float>(frame_->nframe);
        if (new_progress - progress > 0.01)
        {
          progress = new_progress;
          resdata::io::print_progress_bar(progress);
        }
        if ((frame_->time >= t_begin_ && (t_end_ < 0 || frame_->time <= t_end_)) &&
            (dt_ == 0 || std::fmod(frame_->time, dt_) == 0) &&
            (nskip_ == 0 || std::fmod(frnr, nskip_) == 0))
        {
#ifdef DEBUG
          // write frame in text
          std::cout << "Writing frame " << frnr << std::endl;
          std::fstream out("frame_" + std::to_string(frnr) + ".txt", std::ios::out);
          for (int i = 0; i < top_.get_n_atoms(); i++)
          {
            std::cout << "Writing atom " << i << std::endl;
            out << frame_->x[i][0] << " " << frame_->x[i][1] << " " << frame_->x[i][2] << std::endl;
          }
          out.close();
#endif
          float weight = 1.0;
          if (!weights_.empty())
          {
            weight = weights_[frnr];
            weights_sum_ += weight;
          }
          for (std::size_t i = 0; i < frame_same_mat_.size(); i++)
          {
            // #pragma omp parallel for num_threads(std::min(num_threads_, static_cast<int>(frame_same_mat_[i].size())))
            for (std::size_t j = 0; j < frame_same_mat_[i].size(); j++)
              frame_same_mat_[i][j] = 100.;
          }
          for (std::size_t i = 0; i < frame_cross_mat_.size(); i++)
          {
            // #pragma omp parallel for num_threads(std::min(num_threads_, static_cast<int>(frame_cross_mat_[i].size())))
            for (std::size_t j = 0; j < frame_cross_mat_[i].size(); j++)
              frame_cross_mat_[i][j] = 100.;
          }

          // #pragma omp parallel for num_threads(std::min(num_threads_, top_.get_n_mols()))
          for (int i = 0; i < top_.get_n_mols(); i++)
          {
            clear_rvec(xcm_[i]);
            float tm = 0.;
            int from = top_.molblock(i).front();
            int to = top_.molblock(i).back() + 1;
            // for (int ii = top_.molblock(i).front(); ii < top_.molblock(i).back() + 1; ii++)
            for (int ii = from; ii < to; ii++)
            {
              for (int m = 0; (m < DIM); m++)
              {
                xcm_[i][m] += frame_->x[ii][m];
              }
              tm += 1.0;
            }
            for (int m = 0; m < DIM; m++)
            {
              xcm_[i][m] /= tm;
            }
          }
#ifdef DEBUG
          std::fstream xcmout("frame_xcm_" + std::to_string(frnr) + ".txt", std::ios::out);
          for (int i = 0; i < top_.get_n_mols(); i++)
          {
            xcmout << "Molecule " << i << ": ";
            for (int j = 0; j < DIM; j++)
            {
              xcmout << xcm_[i][j] << " ";
            }
            xcmout << std::endl;
          }
          xcmout.close();
#endif
          // #pragma omp parallel for num_threads(std::min(num_threads_, top_.get_n_mols()))
          for (int i = 0; i < top_.get_n_mols(); ++i)
          {
            int mol_id = top_.mol_id(i);
            int res_per_mol = top_.get_res_per_molecule(mol_id);
            for (int j = 0; j < res_per_mol; ++j)
            {
              clear_rvec(res_xcm_[i][j]);
            }
            int from = top_.molblock(i).front();
            int to = top_.molblock(i).back() + 1;
            for (int ii = from, ai = 0; ii < to; ++ii, ++ai)
            {
              int ires = top_.get_local_residue_index(mol_id, ai);
              rvec_inc(res_xcm_[i][ires], frame_->x[ii]);
            }
            // int n_res = top_.get_res_per_molecule(mol_id);
            for (int j = 0; j < top_.get_res_per_molecule(mol_id); ++j)
            {
              for (int k = 0; k < DIM; ++k)
              {
                res_xcm_[i][j][k] /= static_cast<float>(top_.get_atoms_per_residue(mol_id, j));
              }
            }
          }
#ifdef DEBUG
          std::fstream rout("frame_res_xcm_" + std::to_string(frnr) + ".txt", std::ios::out);
          for (int i = 0; i < top_.get_n_mols(); i++)
          {
            rout << "Molecule with " << top_.get_res_per_molecule(top_.mol_id(i)) << " residues" << std::endl;
            rout << "Molecule " << i << ": ";
            for (int j = 0; j < top_.get_res_per_molecule(top_.mol_id(i)); j++)
            {
              rout << res_xcm_[i][j][0] << " " << res_xcm_[i][j][1] << " " << res_xcm_[i][j][2] << std::endl;
            }
          }
          rout.close();
#endif
          begin = std::chrono::steady_clock::now();
          for (int i = 0; i < top_.get_n_mols(); i++)
#ifndef DEBUG
          {
            mol_threads_[i] = std::thread(
                molecule_routine, i, std::cref(top_), xcm_,
                mcut2_, cut_sig_2_, std::cref(density_bins_),
                std::cref(res_xcm_), weight,
                std::ref(frame_same_mat_), std::ref(frame_cross_mat_),
                std::ref(frame_same_mutex_), std::ref(frame_cross_mutex_),
                std::ref(intram_mat_density_),
                std::ref(interm_same_mat_density_),
                std::ref(interm_cross_mat_density_),
                intra_, same_, cross_, std::ref(semaphore_));
          }
          for (auto &thread : mol_threads_)
          {
            thread.join();
          }
#else
            molecule_routine(i, std::cref(top_), xcm_,
                             mcut2_, cut_sig_2_, std::cref(density_bins_),
                             std::cref(res_xcm_), weight,
                             std::ref(frame_same_mat_), std::ref(frame_cross_mat_),
                             std::ref(frame_same_mutex_), std::ref(frame_cross_mutex_),
                             std::ref(intram_mat_density_),
                             std::ref(interm_same_mat_density_),
                             std::ref(interm_cross_mat_density_),
                             intra_, same_, cross_, std::ref(semaphore_));
#endif
          end = std::chrono::steady_clock::now();
          /* calculate the mindist accumulation indices */
          for (int tid = 0; tid < num_threads_; tid++)
          {
            threads_[tid] = std::thread(
                resdata::mindist::mindist_kernel,
                std::cref(same_thread_indices_[tid]),
                std::cref(cross_thread_indices_[tid]),
                weight,
                std::cref(groups_per_molecule),
                std::cref(density_bins_),
                std::cref(n_moltypes),
                std::cref(frame_same_mat_),
                std::ref(frame_same_mutex_),
                std::ref(interm_same_maxcdf_mol_),
                std::cref(cross_index),
                std::cref(frame_cross_mat_),
                std::ref(frame_cross_mutex_),
                std::ref(interm_cross_maxcdf_mol_));
          }
          for (auto &thread : threads_)
            thread.join();
          ++n_x_;
        }
        ++frnr;
      }

      resdata::io::print_progress_bar(1.0);
      std::cout << "number of frames: " << frnr << std::endl;
      std::cout << "frames analyzed: " << n_x_ << std::endl;
    }

    void process_data()
    {
      std::cout << "\nFinished frame-by-frame analysis\n";
      std::cout << "Analyzed " << n_x_ << " frames\n";
      std::cout << "Normalizing data... " << std::endl;
      // normalisations
      float norm = (weights_.empty()) ? 1. / n_x_ : 1. / weights_sum_;

      using ftype_norm = resdata::ftypes::function_traits<decltype(&resdata::density::normalize_histo)>;
      std::function<ftype_norm::signature> f_empty = resdata::ftypes::do_nothing<ftype_norm>();

      std::function<ftype_norm::signature> normalize_intra = (intra_) ? resdata::density::normalize_histo : f_empty;
      std::function<ftype_norm::signature> normalize_inter_same = (same_) ? resdata::density::normalize_histo : f_empty;
      std::function<ftype_norm::signature> normalize_inter_cross = (cross_) ? resdata::density::normalize_histo : f_empty;

      for (std::size_t i = 0; i < top_.get_n_moltypes(); i++)
      {
        for (int ii = 0; ii < top_.get_res_per_molecule(i); ii++)
        {
          for (int jj = ii; jj < top_.get_res_per_molecule(i); jj++)
          {
            float inv_num_mol_same = top_.get_inv_num_mol(i);
            normalize_inter_same(i, ii, jj, norm, inv_num_mol_same, interm_same_maxcdf_mol_);
            normalize_inter_same(i, ii, jj, norm, 1.0, interm_same_mat_density_);
            normalize_intra(i, ii, jj, norm, 1.0, intram_mat_density_);

            float sum = 0.0;
            for (std::size_t k = (same_) ? 0 : resdata::indexing::n_bins(cutoff_); k < resdata::indexing::n_bins(cutoff_); k++)
            {
              sum += dx_ * interm_same_maxcdf_mol_[i][ii][jj][k];
              if (sum > 1.0)
                sum = 1.0;
              interm_same_maxcdf_mol_[i][ii][jj][k] = sum;
            }
            if (same_)
              interm_same_mat_density_[i][jj][ii] = interm_same_mat_density_[i][ii][jj];
            if (same_)
              interm_same_maxcdf_mol_[i][jj][ii] = interm_same_maxcdf_mol_[i][ii][jj];
            if (intra_)
              intram_mat_density_[i][jj][ii] = intram_mat_density_[i][ii][jj];
          }
        }
        for (std::size_t j = i + 1; j < top_.get_n_moltypes() && cross_; j++)
        {
          for (int ii = 0; ii < top_.get_res_per_molecule(i); ii++)
          {
            for (int jj = 0; jj < top_.get_res_per_molecule(j); jj++)
            {
              float inv_num_mol_cross = top_.get_inv_num_mol(i);
              normalize_inter_cross(top_.get_cross_index(i, j), ii, jj, norm, 1.0, interm_cross_mat_density_);
              normalize_inter_cross(top_.get_cross_index(i, j), ii, jj, norm, inv_num_mol_cross, interm_cross_maxcdf_mol_);

              float sum = 0.0;
              for (std::size_t k = (cross_) ? 0 : resdata::indexing::n_bins(cutoff_); k < resdata::indexing::n_bins(cutoff_); k++)
              {
                sum += dx_ * interm_cross_maxcdf_mol_[top_.get_cross_index(i, j)][ii][jj][k];
                if (sum > 1.0)
                  sum = 1.0;
                interm_cross_maxcdf_mol_[top_.get_cross_index(i, j)][ii][jj][k] = sum;
              }
            }
          }
        }
      }
    }

    void write_output(const std::string &output_prefix)
    {
      std::cout << "Writing data... " << std::endl;
      using ftype_write_intra = resdata::ftypes::function_traits<decltype(&resdata::io::f_write_intra)>;
      using ftype_write_inter_same = resdata::ftypes::function_traits<decltype(&resdata::io::f_write_inter_same)>;
      using ftype_write_inter_cross = resdata::ftypes::function_traits<decltype(&resdata::io::f_write_inter_cross)>;
      std::function<ftype_write_intra::signature> write_intra = resdata::ftypes::do_nothing<ftype_write_intra>();
      std::function<ftype_write_inter_same::signature> write_inter_same = resdata::ftypes::do_nothing<ftype_write_inter_same>();
      std::function<ftype_write_inter_cross::signature> write_inter_cross = resdata::ftypes::do_nothing<ftype_write_inter_cross>();

      if (intra_)
      {
#ifdef USE_HDF5
        if (h5_)
          write_intra = resdata::io::f_write_intra_HDF5;
        else
          write_intra = resdata::io::f_write_intra;
#else
        write_intra = resdata::io::f_write_intra;
#endif
      }
      if (same_)
      {
#ifdef USE_HDF5
        if (h5_)
          write_inter_same = resdata::io::f_write_inter_same_HDF5;
        else
          write_inter_same = resdata::io::f_write_inter_same;
#else
        write_inter_same = resdata::io::f_write_inter_same;
#endif
      }
      if (cross_)
      {
#ifdef USE_HDF5
        if (h5_)
          write_inter_cross = resdata::io::f_write_inter_cross_HDF5;
        else
          write_inter_cross = resdata::io::f_write_inter_cross;
#else
        write_inter_cross = resdata::io::f_write_inter_cross;
#endif
      }

      for (std::size_t i = 0; i < top_.get_n_moltypes(); i++)
      {
        std::cout << "Writing data for molecule " << i << "..." << std::endl;
        resdata::io::print_progress_bar(0.0);
        float progress = 0.0, new_progress = 0.0;

        for (int ii = 0; ii < top_.get_res_per_molecule(i); ii++)
        {
          new_progress = static_cast<float>(ii) / static_cast<float>(top_.get_res_per_molecule(i));
          if (new_progress - progress > 0.01)
          {
            progress = new_progress;
            resdata::io::print_progress_bar(progress);
          }
          write_intra(output_prefix, i, ii, density_bins_, top_.get_res_per_molecule(), intram_mat_density_);
          write_inter_same(output_prefix, i, ii, density_bins_, top_.get_res_per_molecule(), interm_same_mat_density_, interm_same_maxcdf_mol_);
        }
        if (cross_)
        {
          for (std::size_t j = i + 1; j < top_.get_n_moltypes(); j++)
          {
            for (int ii = 0; ii < top_.get_res_per_molecule(j); ii++)
            {
              write_inter_cross(output_prefix, i, j, ii, density_bins_, top_.get_res_per_molecule(), top_.get_cross_index(), interm_cross_mat_density_, interm_cross_maxcdf_mol_);
            }
          }
        }
        resdata::io::print_progress_bar(1.0);
        std::cout << "\nFinished!" << std::endl;
      }
    }
  };
} // namespace resdata

#endif // _RESDATA_HPP
