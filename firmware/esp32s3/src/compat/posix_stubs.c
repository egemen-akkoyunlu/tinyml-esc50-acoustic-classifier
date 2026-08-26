/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * POSIX I/O and ESP-IDF symbol stubs for Zephyr + esp-dl compatibility.
 */

#include <sys/types.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

__attribute__((weak)) int open(const char *pathname, int flags, ...)
{
    (void)pathname;
    (void)flags;
    errno = ENOENT;
    return -1;
}

__attribute__((weak)) ssize_t read(int fd, void *buf, size_t count)
{
    (void)fd;
    (void)buf;
    (void)count;
    errno = EIO;
    return -1;
}

__attribute__((weak)) ssize_t write(int fd, const void *buf, size_t count)
{
    (void)fd;
    (void)buf;
    (void)count;
    errno = EIO;
    return -1;
}

__attribute__((weak)) off_t lseek(int fd, off_t offset, int whence)
{
    (void)fd;
    (void)offset;
    (void)whence;
    errno = EINVAL;
    return (off_t)-1;
}

__attribute__((weak)) int close(int fd)
{
    (void)fd;
    errno = EBADF;
    return -1;
}

/* ESP-IDF logging symbol stubs required by precompiled libfbs_model.a */
uint32_t esp_log_timestamp(void)
{
    return (uint32_t)k_uptime_get_32();
}

void esp_log_write(int level, const char *tag, const char *format, ...)
{
    (void)level;
    (void)tag;
    va_list list;
    va_start(list, format);
    vprintk(format, list);
    va_end(list);
}
