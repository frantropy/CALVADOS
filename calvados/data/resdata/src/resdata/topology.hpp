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

#include "function_types.hpp"
// #include "lgmx.hpp"

#include <iostream>
#define LOG(x) std::cout << x << std::endl;

/**
 * NOTES
 * doing decltype(x) & allows for const ref access without copying the data -> should be faster https://stackoverflow.com/a/5424111
 * would be nice to implement ^ where necessary
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
    void erase(std::size_t i)
    {
      partition.erase(std::begin(partition) + i);
    }
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
    std::vector<int> global_residue_indices;
    std::vector<std::vector<int>> residue_indices;
    std::vector<int> res_per_molecule;
    std::vector<int> n_atom_per_molecule;
    std::vector<int> mol_id_;
    // std::vector<int> index_;
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
    void get_global_residue_index(int gai, int &res_gi) const
    {
      res_gi = global_residue_indices[gai];
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
    void get_global_resi_atom(int gai, std::string atomname, int &res_gi) const
    {
      get_atom_name(gai, atomname);
      get_global_residue_index(gai, res_gi);
    }
    // void set_topol_pbc(t_pbc *pbc)
    // {
    //   if (pbc_ != nullptr)
    //   {
    //     free(pbc_);
    //   }
    //   this->pbc_ = (t_pbc *)malloc(sizeof(t_pbc));
    //   this->pbc_ = pbc;
    // }
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
          for (auto row : cross_index)
          {
            for (auto col : row)
            {
              if (col > ci_max)
              {
                ci_max = col;
              }
            }
          }

          for (int j = 0; j < cross_index.size(); ++j)
          {
            cross_index[j].push_back(ci_max++);
          }
          cross_index.push_back(std::vector<int>(1, ci_max));

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

        int global_res_index = (global_residue_indices.empty()) ? 0 : global_residue_indices.back() + 1;

        for (int ri = 0; ri < residue_names.size(); ++ri)
        {
          global_residue_indices.push_back(global_res_index + ri);
        }

        for (int j = 0; j < atom_names.size(); ++j)
        {
          global_atom_names.push_back(atom_names[j]);
        }
      }
    }
    int get_n_atoms() const
    {
      return n_atoms;
    }
    // int get_n_atoms( int i, bool type_wise = false ) const
    // {
    //   if (type_wise) i = mol_id_[i];
    //   return n_atom_per_molecule[i];
    // }
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

    void print_topology_summary(const std::string &path) const
    {
      std::ofstream outFile(path);
      if (!outFile.is_open())
      {
        throw std::runtime_error("Failed to open file for writing: " + path);
      }

      outFile << "Topology Summary:\n";
      outFile << "--------------------\n";
      outFile << "Total number of molecules: " << get_n_mols() << "\n";
      outFile << "Total number of unique molecule types: " << get_n_moltypes() << "\n";
      outFile << "Total number of atoms: " << get_n_atoms() << "\n\n";

      for (std::size_t moltype = 0; moltype < n_mols.size(); ++moltype)
      {
        outFile << "Molecule Type " << moltype << ":\n";
        outFile << "  Number of molecules: " << n_mols[moltype] << "\n";
        outFile << "  Number of atoms per molecule: " << n_atom_per_molecule[moltype] << "\n";
        outFile << "  Number of residues per molecule: " << res_per_molecule[moltype] << "\n";
        outFile << "  Residues:\n";

        const auto &resnames = residue_names[moltype];
        const auto &atomsperres = atoms_per_residue[moltype];
        std::size_t atom_counter = 0;

        for (std::size_t resi = 0; resi < atomsperres.size(); ++resi)
        {
          if (atom_counter < resnames.size())
          {
            outFile << "    Residue " << resi << " (" << resnames[atom_counter] << "): "
                    << atomsperres[resi] << " atoms\n";
          }
          else
          {
            outFile << "    Residue " << resi << " (UNKNOWN): " << atomsperres[resi] << " atoms\n";
          }
          atom_counter += atomsperres[resi];
        }
        outFile << "\n";
      }

      outFile << "--------------------\n";
      if (pbc_ != nullptr)
      {
        outFile << "Periodic Boundary Conditions:\n";
        outFile << "  Type (int value): " << static_cast<int>(pbc_->pbcType) << "\n";
        outFile << "  Box matrix:\n";
        for (int i = 0; i < 3; ++i)
        {
          outFile << "    [" << pbc_->box[i][0] << ", "
                  << pbc_->box[i][1] << ", "
                  << pbc_->box[i][2] << "]\n";
        }
      }
      else
      {
        outFile << "No periodic boundary conditions set.\n";
      }
      // print the partitioning
      outFile << "Molecule Partitioning:\n";
      for (std::size_t i = 0; i < mols.size(); ++i)
      {
        outFile << "  Partition " << i << ": ";
        for (const auto &atom : mols[i])
        {
          outFile << atom << " ";
        }
        outFile << "\n";
      }
      outFile << "--------------------\n";
      outFile << "Cross Index:\n";
      for (std::size_t i = 0; i < cross_index.size(); ++i)
      {
        outFile << "  Cross Index for molecule type " << i << ": ";
        for (const auto &index : cross_index[i])
        {
          outFile << index << " ";
        }
        outFile << "\n";
      }
      outFile << "--------------------\n";
      outFile << "Residue local indices:\n";
      for (std::size_t i = 0; i < get_local_residue_index().size(); ++i)
      {
        outFile << "  Residue local index for molecule type " << i << ": ";
        for (const auto &index : get_local_residue_index(i))
        {
          outFile << index << " ";
        }
        outFile << "\n";
      }

      outFile.close();
    }

    void apply_index(const std::vector<int> index)
    {
      if (index.empty()) return;
      std::vector<int> to_remove;
      for (int mbi = 0; mbi < mols.size(); mbi++)
      {
        for (int j = 0; j < mols[mbi].size(); j++)
        {
          if (std::find(std::begin(index), std::end(index), mols[mbi][j]) == std::end(index))
          {
            to_remove.push_back(mbi);
            break;
          }
        }
      }
      std::reverse(std::begin(to_remove), std::end(to_remove));
      for (auto i : to_remove)
      {
        mols.erase(i);
        int id = mol_id_[i];
        mol_id_.erase(std::begin(mol_id_) + i);
        n_mols[id]--;
        n_atoms -= n_atom_per_molecule[id];

        if (n_mols[id] < 1)
        {
          atom_names.erase(std::begin(atom_names) + i);
          residue_names.erase(std::begin(residue_names) + i);
          residue_indices.erase(std::begin(residue_indices) + i);
          res_per_molecule.erase(std::begin(res_per_molecule) + id);
          n_atom_per_molecule.erase(std::begin(n_atom_per_molecule) + id);
          atoms_per_residue.erase(std::begin(atoms_per_residue) + id);
          n_mols.erase(std::begin(n_mols) + id);
          std::string molecule_name;
          for (const auto &it : mol_name_to_id)
          {
            if (it.second == id)
            {
              molecule_name = it.first;
              break;
            }
          }
          mol_name_to_id.erase(molecule_name);

          for (int j = 0; j < mol_id_.size(); ++j)
          {
            mol_id_[j]--;
          }
          for (auto &it : mol_name_to_id)
          {
            it.second--;
          }
        }
      }
    }
  };
} // namespace resdata::topology

#endif // _TOPOLOGY_CLASS
