#ifndef _RESDATA_IO_HPP
#define _RESDATA_IO_HPP

#include <gromacs/trajectoryanalysis/topologyinformation.h>
#include <gromacs/fileio/tpxio.h>
#include <gromacs/math/vec.h>
#include <gromacs/pbcutil/pbc.h>
#include <gromacs/fileio/confio.h>
#include <gromacs/topology/index.h>

#include <filesystem>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iomanip>
#include <regex>
#include <tuple>

#define COUT_FLOAT_PREC6 std::fixed << std::setprecision(6)

namespace resdata::io
{

std::tuple<
  std::vector<std::vector<float>>, std::string, std::vector<int>, std::vector<std::string>,
  std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>, std::vector<std::vector<int>>
> read_simple_topology( const std::string &path )
{
  bool box_found = false, molecule_found = false, pbc_found = false;
  std::vector<std::vector<float>> box;
  std::string pbc;
  std::vector<int> n_mol;
  std::vector<std::string> molecules;
  std::vector<std::vector<std::string>> atom_names;
  std::vector<std::vector<std::string>> residue_names;
  std::vector<std::vector<int>> residue_numbers;

  std::ifstream infile(path);
  if (!infile.good())
  {
    std::string errorMessage = "Cannot find the indicated topology file";
    throw std::runtime_error(errorMessage.c_str());
  }
  std::string line;
  while (std::getline(infile, line))
  {
    if (line == "")
    {
      printf("Detected empty line. Skipping...\n");
      continue;
    }
    if (line.find("box") != std::string::npos)
    {
      if (box_found)
      {
        std::string errorMessage = "Multiple box definitions found in the topology file";
        throw std::runtime_error(errorMessage.c_str());
      }
      box_found = true;
      std::istringstream iss(line);
      std::string value;
      iss >> value;
      for (int i = 0; i < 3; ++i)
      {
        box.push_back(std::vector<float>(3));
        iss >> box[i][0] >> box[i][1] >> box[i][2];
      }
    }
    if (line.find("molecule") != std::string::npos)
    {
      molecule_found = true;
      std::istringstream iss(line);
      std::string value;
      iss >> value;
      std::string molecule_name;
      iss >> molecule_name;
      molecules.push_back(molecule_name);
      atom_names.push_back(std::vector<std::string>());
      residue_names.push_back(std::vector<std::string>());
      residue_numbers.push_back(std::vector<int>());
      while (iss >> value)
      {
        if (std::isdigit(value[0]))
        {
          n_mol.push_back(std::stoi(value));
          break;
        }
        else
        {
          // split the value at the underscore
          std::size_t pos_res = value.find('_');
          std::size_t pos_num = value.find('-');
          if (pos_res != std::string::npos || pos_num != std::string::npos)
          {
            std::string atom_name = value.substr(0, pos_res);
            std::string residue_name = value.substr(pos_res + 1, pos_num - pos_res - 1);
            std::string resnum_str = value.substr(pos_num + 1);
            int resnum = std::stoi(resnum_str);
            residue_names.back().push_back(residue_name);
            atom_names.back().push_back(atom_name);
            residue_numbers.back().push_back(resnum);
          }
          else
          {
            residue_names.back().push_back("XXX");
            atom_names.back().push_back(value);
            residue_numbers.back().push_back(1);
          }
        }
      }
    }
    if (line.find("pbc") != std::string::npos)
    {
      if (pbc_found)
      {
        std::string errorMessage = "Multiple PBC definitions found in the topology file";
        throw std::runtime_error(errorMessage.c_str());
      }
      pbc_found = true;
      std::istringstream iss(line);
      std::string value;
      iss >> value;
      iss >> pbc;
      printf("Found PBC value: %s\n", pbc.c_str());
    }
  }

  if (!box_found)
  {
    std::string errorMessage = "No box found in the topology file";
    throw std::runtime_error(errorMessage.c_str());
  }
  if (!molecule_found)
  {
    std::string errorMessage = "No molecule found in the topology file";
    throw std::runtime_error(errorMessage.c_str());
  }
  if (!pbc_found)
  {
    pbc = std::string("unset");
  }

  return std::make_tuple(box, pbc, n_mol, molecules, atom_names, residue_names, residue_numbers);
}

std::vector<float> read_weights_file( const std::string &path )
{
  std::ifstream infile(path);
  if (!infile.good())
  {
    std::string errorMessage = "Cannot find the indicated weights file";
    throw std::runtime_error(errorMessage.c_str());
  }
  std::vector<float> w;

  std::string line;
  while ( std::getline(infile, line) )
  {
    std::string value;
    std::istringstream iss(line);
    if (line == "")
    {
      printf("Detected empty line. Skipping...\n");
      continue;
    }
    iss >> value;
    w.push_back(std::stod(value));
  }

  if (w.size() == 0)
  {
    std::string errorMessage = "The weights file is empty";
    throw std::runtime_error(errorMessage.c_str());
  }

  for ( std::size_t i = 0; i < w.size(); i++ )
  {
    if (w[i] < 0)
    {
      std::string errorMessage = "The weights file contains negative values";
      throw std::runtime_error(errorMessage.c_str());
    }
  }

  return w;
}

void f_write_intra(const std::string &output_prefix,
  std::size_t i, int ii, const std::vector<float> &density_bins, const std::vector<int> &natmol2,
  const std::vector<std::vector<std::vector<std::vector<float>>>> &intram_mat_density
)
{
  std::filesystem::path ffh_intra = output_prefix + "intra_mol_" + std::to_string(i + 1) + "_" + std::to_string(i + 1) + "_aa_" + std::to_string(ii + 1) + ".dat";
  std::ofstream fp_intra(ffh_intra);
  for ( std::size_t k = 0; k < density_bins.size(); k++ )
  {
    fp_intra << COUT_FLOAT_PREC6 << density_bins[k];
    for (int jj = 0; jj < natmol2[i]; jj++)
    {
      fp_intra << " " << COUT_FLOAT_PREC6 << intram_mat_density[i][ii][jj][k];
    }
    fp_intra << "\n";
  }

  fp_intra.close();
}

void f_write_inter_same(const std::string &output_prefix,
  std::size_t i, int ii, const std::vector<float> &density_bins, const std::vector<int> &natmol2,
  const std::vector<std::vector<std::vector<std::vector<float>>>> &interm_same_mat_density,
  const std::vector<std::vector<std::vector<std::vector<float>>>> &interm_same_maxcdf_mol
)
{
  std::filesystem::path ffh_inter = output_prefix + "inter_mol_" + std::to_string(i + 1) + "_" + std::to_string(i + 1) + "_aa_" + std::to_string(ii + 1) + ".dat";
  std::ofstream fp_inter(ffh_inter);
  std::filesystem::path ffh_inter_cum = output_prefix + "inter_mol_c_" + std::to_string(i + 1) + "_" + std::to_string(i + 1) + "_aa_" + std::to_string(ii + 1) + ".dat";
  std::ofstream fp_inter_cum(ffh_inter_cum);
  for ( std::size_t k = 0; k < density_bins.size(); k++ )
  {
    fp_inter << COUT_FLOAT_PREC6 << density_bins[k];
    fp_inter_cum << COUT_FLOAT_PREC6 << density_bins[k];
    for (int jj = 0; jj < natmol2[i]; jj++)
    {
      fp_inter << " " << COUT_FLOAT_PREC6 << interm_same_mat_density[i][ii][jj][k];
      fp_inter_cum << " " << COUT_FLOAT_PREC6 << interm_same_maxcdf_mol[i][ii][jj][k];
    }
    fp_inter << "\n";
    fp_inter_cum << "\n";
  }
  fp_inter.close();
  fp_inter_cum.close();
}

void f_write_inter_cross(const std::string &output_prefix,
  std::size_t i, std::size_t j, int ii, const std::vector<float> &density_bins, const std::vector<int> &natmol2,
  const std::vector<std::vector<int>> &cross_index,
  const std::vector<std::vector<std::vector<std::vector<float>>>> &interm_cross_mat_density,
  const std::vector<std::vector<std::vector<std::vector<float>>>> &interm_cross_maxcdf_mol
)
{
  std::filesystem::path ffh = output_prefix + "inter_mol_" + std::to_string(i + 1) + "_" + std::to_string(j + 1) + "_aa_" + std::to_string(ii + 1) + ".dat";
  std::ofstream fp(ffh);
  std::filesystem::path ffh_cum = output_prefix + "inter_mol_c_" + std::to_string(i + 1) + "_" + std::to_string(j + 1) + "_aa_" + std::to_string(ii + 1) + ".dat";
  std::ofstream fp_cum(ffh_cum);
  for ( std::size_t k = 0; k < interm_cross_mat_density[cross_index[i][j]][ii][0].size(); k++ )
  {
    fp << std::fixed << std::setprecision(7) << density_bins[k];
    fp_cum << std::fixed << std::setprecision(7) << density_bins[k];
    for (int jj = 0; jj < natmol2[j]; jj++)
    {
      fp << " " << std::fixed << std::setprecision(7) << interm_cross_mat_density[cross_index[i][j]][ii][jj][k];
      fp_cum << " " << std::fixed << std::setprecision(7) << interm_cross_maxcdf_mol[cross_index[i][j]][ii][jj][k];
    }
    fp << "\n";
    fp_cum << "\n";
  }
  fp.close();
  fp_cum.close();
}

std::tuple<
  std::vector<std::string>, std::vector<std::vector<int>>
> read_index_file( const std::string &path )
{
  std::ifstream infile(path);
  if (!infile.good())
  {
    std::string errorMessage = "Cannot find the indicated index file";
    throw std::runtime_error(errorMessage.c_str());
  }

  std::vector<std::string> index_names;
  std::vector<std::vector<int>> index;
  std::string line;

  while( std::getline(infile, line) )
  {
    if (line == "") continue;
    std::istringstream iss(line);
    std::string value;
    if ( line.find(";") != std::string::npos ) continue;
    if ( line.find("[") != std::string::npos )
    {
      std::string index_name;
      std::string trash;
      iss >> trash;
      iss >> index_name;
      index_names.push_back(index_name);
      index.push_back(std::vector<int>());
      continue;
    }
    while (iss >> value)
    {
      index.back().push_back(std::stoi(value)-1);
    }
  }

  return {index_names, index};
}

/**
   * @brief Print a progress bar to the standard output
   * 
   * Taken from https://stackoverflow.com/a/36315819
   * 
   * @param percentage
  */
void print_progress_bar(float percentage)
{
  constexpr std::size_t PROGRESS_BAR_LENGTH = 60;
  constexpr char PROGRESS_BAR[] = "############################################################";
  
  int val = (int) (percentage * 100);
  int lpad = (int) (percentage * PROGRESS_BAR_LENGTH);
  int rpad = PROGRESS_BAR_LENGTH - lpad;
  
  printf("\r%3d%% [%.*s%*s]", val, lpad, PROGRESS_BAR, rpad, "");
  fflush(stdout);
}

} // namespace resdata::io

#endif // _CMDATA_IO_HPP

