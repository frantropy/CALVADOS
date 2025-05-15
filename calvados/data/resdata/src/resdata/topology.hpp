#ifndef _TOPOLOGY_CLASS
#define _TOPOLOGY_CLASS

#include <gromacs/trajectoryanalysis/topologyinformation.h>
#include <gromacs/math/vec.h>
#include <gromacs/pbcutil/pbc.h>
#include <gromacs/fileio/tpxio.h>
#include <gromacs/fileio/confio.h>
#include <gromacs/topology/index.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "function_types.hpp"

#include <iostream>

/**
 * NOTES
 * doing decltype(x) & allows for const ref access without copying the data -> should be faster https://stackoverflow.com/a/5424111
 * would be nice to implement ^ where necessary
 * TODO:
 *  - remove all the mol_id bools as it is ugly and bad
 */

namespace resdata::topology
{
  class RangePartitioning
  {
  private:
    std::vector<std::vector<int>> partition;
    std::vector<int> partition_index;

  public:
    RangePartitioning() {}
    RangePartitioning(std::vector<std::vector<int>> partition) : partition(partition) {}
    RangePartitioning(int end)
    {
      partition.push_back(std::vector<int>(end));
      std::fill(std::begin(partition[0]), std::end(partition[0]), end);
    }
    void add_partition(std::vector<int> partition)
    {
      this->partition.push_back(partition);
      partition_index.push_back(partition_index.size());
    }
    void add_partition(int n)
    {
      std::vector<int> new_partition(n);
      int from = partition.empty() ? 0 : partition.back().back() + 1;
      // std::iota(std::begin(new_partition), std::end(new_partition), from);
      for (int i = from, index = 0; i < from + n; ++i, ++index)
      {
        new_partition[index] = i;
      }
      partition.push_back(new_partition);
      partition_index.push_back(partition_index.size());
    }
    int get_partition_index(int i) const
    {
      return partition_index[i];
    }
    std::size_t size() const
    {
      return partition.size();
    }
    const std::vector<int> &operator[](std::size_t i) const
    {
      return partition[i];
    }
    std::vector<int> &operator[](std::size_t i)
    {
      return partition[i];
    }
    auto begin() const { return partition.begin(); }
    auto end() const { return partition.end(); }
    auto front() const { return partition.front(); }
    auto back() const { return partition.back(); }
    bool empty() const { return partition.empty(); }
    void erase(std::size_t i) { partition.erase(std::begin(partition) + i); }
  };

  class Topology
  {
  private:
    t_pbc *pbc_ = nullptr;
    RangePartitioning mols;
    std::vector<std::vector<std::string>> atom_names;
    std::vector<std::string> global_atom_names;
    std::vector<std::vector<int>> atoms_per_residue;
    std::vector<float> inv_num_mol;
    std::vector<std::vector<std::string>> residue_names;
    std::vector<std::vector<int>> residue_indices;
    std::vector<int> res_per_molecule;
    std::vector<int> n_atom_per_molecule;
    std::vector<int> mol_id_;
    std::vector<int> n_mols;
    std::vector<std::vector<int>> cross_index;
    int n_atoms = 0;
    std::unordered_map<std::string, int> mol_name_to_id;

  public:
    Topology() {}
    ~Topology()
    {
      unset_pbc();
    }
    void set_box(const matrix &box)
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          pbc_->box[i][j] = box[i][j];
        }
      }
    }
    int get_n_mols(int mi, bool type_wise = false) const
    {
      if (type_wise)
        mi = mol_id_[mi];
      return n_mols[mi];
    }
    int get_n_atoms_per_molecule(int i, bool type_wise = false) const
    {
      if (type_wise)
        i = mol_id_[i];
      return n_atom_per_molecule[i];
    }
    int get_n_mols() const
    {
      return mols.size();
    }
    int get_n_moltypes() const
    {
      return n_mols.size();
    }
    const std::vector<int> get_unique_molecules() const
    {
      return n_mols;
    }
    int get_atoms_per_residue(int mi, int i) const
    {
      return atoms_per_residue[mi][i];
    }
    void get_atom_name(int gai, std::string &atomname) const
    {
      atomname = global_atom_names[gai];
    }
    std::vector<std::vector<int>> get_local_residue_index() const
    {
      return residue_indices;
    }
    std::vector<int> get_local_residue_index(int mi) const
    {
      return residue_indices[mi];
    }
    int get_local_residue_index(int mi, int i)
    {
      return residue_indices[mi][i];
    }
    void set_topol_pbc(const PbcType &pbcType, const matrix &box)
    {
      if (pbc_ != nullptr)
      {
        free(pbc_);
      }
      pbc_ = (t_pbc *)malloc(sizeof(t_pbc));
      set_pbc(pbc_, pbcType, box);
    }
    std::vector<int> get_n_atoms_per_molecule() const
    {
      return n_atom_per_molecule;
    }
    void set_topol_pbc(const std::string &pbcType, const matrix &box)
    {
      PbcType mapped_pbcType = resdata::dtypes::pbc_type_map.at(pbcType);
      set_topol_pbc(mapped_pbcType, box);
    }
    void set_topol_pbc(const std::string &pbcType, const std::vector<std::vector<float>> &box)
    {
      matrix mapped_box;
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          mapped_box[i][j] = box[i][j];
        }
      }
      set_topol_pbc(pbcType, mapped_box);
    }
    const float *get_box() const
    {
      return &pbc_->box[0][0];
    }
    t_pbc *get_pbc() const
    {
      return pbc_;
    }
    PbcType get_pbc_type() const
    {
      return pbc_->pbcType;
    }
    RangePartitioning molblock() const
    {
      return mols;
    }
    std::vector<int> molblock(int i) const
    {
      return mols[i];
    }
    std::vector<int> get_molblock_indices() const
    {
      std::vector<int> indices(mols.size());
      std::iota(std::begin(indices), std::end(indices), 0);
      return indices;
    }
    std::vector<std::vector<int>> build_cross_index( const std::vector<int> &mid )
    {
      std::vector<std::vector<int>> cross_idx(mid.size(), std::vector<int>(mid.size()));
      int cross_id = 0;
      for ( int i = 0; i < mid.size(); ++i )
      {
        for ( int j = i; j < mid.size(); ++j )
        {
          cross_idx[i][j] = cross_id;
          cross_idx[j][i] = cross_id;
          ++cross_id;
        }
      }

      return cross_idx;
    }
    /**
     * @todo Add proper cross index generation
     */
    void add_molecule(
        const std::string &molecule_name, const std::vector<std::string> &atom_names,
        const std::vector<std::string> &residue_names, const std::vector<int> &residue_indices,
        int n = 1)
    {
      if (atom_names.size() != residue_names.size())
      {
        std::string errorMessage = "Atom names and residue names must have the same size";
        throw std::runtime_error(errorMessage.c_str());
      }

      int id;
      for (int i = 0; i < n; ++i)
      {
        if (mol_name_to_id.find(molecule_name) == std::end(mol_name_to_id))
        {

          int id_prev = -1;
          for (auto &it : mol_name_to_id)
          {
            id_prev = (id_prev > it.second) ? id_prev : it.second;
          }
          id = id_prev + 1;

          mol_id_.push_back(id);
          int ci_max = 0;
          // for (auto row : cross_index)
          // {
          //   for (auto col : row)
          //   {
          //     if (col > ci_max)
          //     {
          //       ci_max = col;
          //     }
          //   }
          // }

          // for (int j = 0; j < cross_index.size(); ++j)
          // {
          //   cross_index[j].push_back(ci_max++);
          // }
          // cross_index.push_back(std::vector<int>(1, ci_max));

          mol_name_to_id.insert({molecule_name, id});
          std::vector<int> mapped_residue_indices(residue_indices.size());
          for (int j = 0, lri = 0, prev_ridx = residue_indices[0]; j < residue_indices.size(); ++j)
          {
            if (prev_ridx != residue_indices[j])
            {
              lri++;
              prev_ridx = residue_indices[j];
            }
            mapped_residue_indices[j] = lri;
          }
          this->atom_names.push_back(atom_names);
          this->residue_names.push_back(residue_names);
          this->residue_indices.push_back(mapped_residue_indices);
          this->n_atom_per_molecule.push_back(atom_names.size());

          n_mols.push_back(1);
          inv_num_mol.push_back(1.0f);

          int prev_resid = mapped_residue_indices[0];
          int rc = 0;
          int atom_per_res = 0;

          atoms_per_residue.push_back(std::vector<int>());
          for (auto r : mapped_residue_indices)
          {
            if (r == prev_resid)
            {
              atom_per_res++;
              continue;
            }
            atoms_per_residue[id].push_back(atom_per_res);
            rc++;
            prev_resid = r;
            atom_per_res = 1;
          }
          atoms_per_residue[id].push_back(atom_per_res);
          rc++;
          res_per_molecule.push_back(rc);
        }
        else
        {
          id = mol_name_to_id[molecule_name];

          n_mols[id] += n;

          mol_id_.push_back(id);
          inv_num_mol[id] = static_cast<float>(n_mols[id]) / static_cast<float>(n);
        }
        n_atoms += atom_names.size();
        mols.add_partition(atom_names.size());

        // int global_res_index = (global_residue_indices.empty()) ? 0 : global_residue_indices.back() + 1;

        // for (int ri = 0; ri < residue_names.size(); ++ri)
        // {
        //   global_residue_indices.push_back(global_res_index + ri);
        // }

        for (int j = 0; j < atom_names.size(); ++j)
        {
          global_atom_names.push_back(atom_names[j]);
        }
      }

      cross_index = build_cross_index(mol_id_);
    }
    int get_n_atoms() const
    {
      return n_atoms;
    }
    const std::vector<std::vector<int>> get_cross_index() const
    {
      return cross_index;
    }
    int get_cross_index(int i, int j, bool type_wise = false) const
    {
      if (type_wise)
      {
        i = mol_id_[i];
        j = mol_id_[j];
      }
      return cross_index[i][j];
    }
    const std::vector<int> mol_id() const
    {
      return mol_id_;
    }
    const float get_inv_num_mol(int i) const
    {
      return inv_num_mol[i];
    }
    const std::vector<float> get_inv_num_mol() const
    {
      return inv_num_mol;
    }
    const int mol_id(int i) const
    {
      return mol_id_[i];
    }
    int get_res_per_molecule(int i, bool type_wise = false) const
    {
      if (type_wise)
        i = mol_id_[i];
      return res_per_molecule[i];
    }
    std::vector<int> get_res_per_molecule() const
    {
      return res_per_molecule;
    }
    void unset_pbc()
    {
      if (pbc_ != nullptr)
      {
        free(pbc_);
      }
    }

    void apply_index( const std::vector<int> &index )
    {
      if (index.empty()) return;

      std::unordered_set<int> index_set(index.begin(), index.end());

      RangePartitioning new_mols;
      std::vector<std::vector<std::string>> new_atom_names;
      std::vector<std::string> new_global_atom_names;
      std::vector<std::vector<int>> new_atoms_per_residue;
      std::vector<float> new_inv_num_mol;
      std::vector<std::vector<std::string>> new_residue_names;
      std::vector<std::vector<int>> new_residue_indices;
      std::vector<int> new_res_per_molecule;
      std::vector<int> new_n_atom_per_molecule;
      std::vector<int> new_mol_id_;
      std::vector<int> new_n_mols(n_mols.size(), 0);
      std::vector<int> index_map(n_atoms, -1); // maps old atom index -> new atom index
      int new_atom_index = 0;

      int global_atom_count = 0;

      std::vector<int> local_properties_updated;

      for (size_t mol_idx = 0; mol_idx < mols.size(); ++mol_idx)
      {
        const auto &mol = mols[mol_idx];
        std::vector<int> filtered_atoms;

        for (int atom : mol)
        {
          if (index_set.count(atom))
          {
            filtered_atoms.push_back(atom);
            index_map[atom] = new_atom_index++;
          }
        }

        if (filtered_atoms.empty()) continue;

        new_mols.add_partition(filtered_atoms);

        int mol_type = mol_id_[mol_idx];
        new_mol_id_.push_back(mol_type);
        new_n_mols[mol_type]++;

        std::vector<std::string> mol_atom_names;
        std::vector<std::string> mol_residue_names;
        std::vector<int> mol_residue_indices;
        std::vector<int> mol_atoms_per_res;
        const auto &res_idx = residue_indices[mol_type];
        const auto &res_names = residue_names[mol_type];
        const auto &atom_names_list = atom_names[mol_type];

        int prev_res = -1;
        int atom_per_res_count = 0;

        bool type_already_updated = std::end(local_properties_updated) != std::find(
          std::begin(local_properties_updated), std::end(local_properties_updated), mol_type
        );
        if (type_already_updated) continue;
        for (int atom_local_idx = 0; atom_local_idx < filtered_atoms.size(); ++atom_local_idx)
        {
          int global_atom_idx = filtered_atoms[atom_local_idx];
          int local_filtered_atom_idx = global_atom_idx - mol.front();

          mol_atom_names.push_back(atom_names_list[local_filtered_atom_idx]);
          new_global_atom_names.push_back(atom_names_list[local_filtered_atom_idx]);

          int local_res_idx = res_idx[local_filtered_atom_idx];
          mol_residue_indices.push_back(local_res_idx);
          mol_residue_names.push_back(res_names[local_res_idx]);

          if (prev_res == -1 || local_res_idx != prev_res)
          {
            if (atom_per_res_count > 0)
              mol_atoms_per_res.push_back(atom_per_res_count);
            atom_per_res_count = 1;
            prev_res = local_res_idx;
          }
          else
          {
            atom_per_res_count++;
          }
        }
        if (atom_per_res_count > 0) mol_atoms_per_res.push_back(atom_per_res_count);

        // map residue indices to new indices
        std::vector<int> unique_res_indices;
        for ( int i = 0; i < mol_residue_indices.size(); ++i )
        {
          if ( std::find( std::begin(unique_res_indices), std::end(unique_res_indices), mol_residue_indices[i] ) == std::end(unique_res_indices) )
          {
            unique_res_indices.push_back(mol_residue_indices[i]);
          }
        }
        std::vector<int> unique_res_indices_new(unique_res_indices.size());
        std::iota(std::begin(unique_res_indices_new), std::end(unique_res_indices_new), 0);
        std::unordered_map<int, int> res2res_map;
        for ( int i = 0; i < unique_res_indices.size(); ++i )
        {
          res2res_map[unique_res_indices[i]] = unique_res_indices_new[i];
        }
        std::vector<int> mol_residue_indices_mapped(unique_res_indices.size());
        for (size_t j = 0; j < unique_res_indices.size(); ++j)
        {
          mol_residue_indices_mapped[j] = res2res_map[mol_residue_indices[j]];
        }

        new_atom_names.push_back(mol_atom_names);
        new_residue_names.push_back(mol_residue_names);
        new_residue_indices.push_back(mol_residue_indices_mapped);
        new_atoms_per_residue.push_back(mol_atoms_per_res);
        new_res_per_molecule.push_back(mol_atoms_per_res.size());
        new_n_atom_per_molecule.push_back(mol_atom_names.size());
      }

      new_inv_num_mol.resize(new_n_mols.size());
      for (size_t i = 0; i < new_n_mols.size(); ++i)
      {
        new_inv_num_mol[i] = new_n_mols[i] > 0 ? 1.0f / static_cast<float>(new_n_mols[i]) : 0.0f;
      }

      std::vector<std::vector<int>> cidx = build_cross_index(new_mol_id_);

      mols = std::move(new_mols);
      atom_names = std::move(new_atom_names);
      global_atom_names = std::move(new_global_atom_names);
      atoms_per_residue = std::move(new_atoms_per_residue);
      inv_num_mol = std::move(new_inv_num_mol);
      residue_names = std::move(new_residue_names);
      residue_indices = std::move(new_residue_indices);
      res_per_molecule = std::move(new_res_per_molecule);
      n_atom_per_molecule = std::move(new_n_atom_per_molecule);
      mol_id_ = std::move(new_mol_id_);
      n_mols = std::move(new_n_mols);
      n_atoms = index.size();
      cross_index = std::move(cidx);
    }
  };
} // namespace resdata::topology

#endif // _TOPOLOGY_CLASS
